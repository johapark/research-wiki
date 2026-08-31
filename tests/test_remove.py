"""Retracting a paper.

The policy this file exists to pin: **generated text is removed, authored text
is reported.** A back-link bullet and an `index.md` entry were written by
`promote`; a sentence on a synthesis page citing `[[stem#slug]]` was written by
a human and has passed both gates. There is no safe rewrite rule for the
second — stripping the citation leaves an unsupported claim, deleting the
sentence can remove a conclusion several papers jointly carried — so `remove`
lists those and edits none of them.

The prose tests assert on **file bytes**, not on the absence of an error. That
is deliberate: "did not crash" would pass even if the page were rewritten.

Hermetic: tmp wiki, no PDFs, no DB writes to the real state.db, no LLM.
"""

from __future__ import annotations

import json

import pytest

from researchwiki import mutation as mut
from researchwiki import removal
from researchwiki.tasks import remove as cli

STEM = "smith-2024-a-paper-about-things"


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    """A wiki with the full range of references to STEM."""
    from researchwiki import paths

    (tmp_path / "wiki" / "compbio").mkdir(parents=True)
    (tmp_path / "wiki" / "synthesis").mkdir()
    (tmp_path / "wiki" / "concepts").mkdir()
    (tmp_path / "papers").mkdir()

    # The paper itself.
    (tmp_path / "wiki" / "compbio" / f"{STEM}.md").write_text(
        f"---\ntype: paper\n---\n\n# A paper\n\n## Summary\nbody\n", encoding="utf-8")
    (tmp_path / "papers" / f"{STEM}.pdf").write_bytes(b"%PDF-1.4\n")

    # A citing paper with a generated back-link bullet.
    (tmp_path / "wiki" / "compbio" / "jones-2025-another.md").write_text(
        "---\ntype: paper\n---\n\n# Another\n\n## Related Papers\n\n"
        f"- [[compbio/{STEM}]] — cites this paper (auto-added; refine)\n"
        "- [[compbio/keep-2020-unrelated]] — topically related\n",
        encoding="utf-8")

    # A synthesis page with an authored claim anchor. MUST NOT be edited.
    (tmp_path / "wiki" / "synthesis" / "a-field-map.md").write_text(
        "---\ntype: synthesis\ncategory: [compbio]\n---\n\n"
        "## Question\nWhat?\n\n## Short answer\n"
        f"Three assemblers agree on the bubble count[^smith].\n\n"
        "## References\n\n"
        f"[^smith]: [[compbio/{STEM}]] — Smith 2024\n",
        encoding="utf-8")

    # A concept hub: generated spoke registry, authored Definition.
    (tmp_path / "wiki" / "concepts" / "pangenome.md").write_text(
        "---\ntype: concept\nconcept_span: 2\n"
        f'referenced_papers: ["[[compbio/{STEM}]]", "[[ai/other-2023-thing]]"]\n'
        "---\n\n## Definition\n\nA pangenome is a thing.\n\n"
        "## How it appears across the corpus\n\n"
        f"- [[compbio/{STEM}]] — builds one from 668 assemblies\n"
        "- [[ai/other-2023-thing]] — uses one downstream\n",
        encoding="utf-8")

    (tmp_path / "wiki" / "index.md").write_text(
        "# index.md\n\n## compbio\n\n"
        f"- [[compbio/{STEM}]] — **Smith 2024** — *A paper*: does a thing.\n"
        "- [[compbio/jones-2025-another]] — **Jones 2025** — *Another*: other thing.\n",
        encoding="utf-8")
    (tmp_path / "wiki" / "log.md").write_text("# log\n\n", encoding="utf-8")

    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(mut, "mutation_dir", lambda: tmp_path / ".mutation")
    monkeypatch.setattr(removal, "_count_db_rows", lambda stem: {})
    monkeypatch.setattr(removal, "_count_edges", lambda stem: 0)
    monkeypatch.setattr(removal, "_delete_db_rows", lambda stem: {})
    monkeypatch.setattr(removal, "_delete_edges", lambda stem: 0)
    monkeypatch.delenv("RW_MUTATION_JOURNAL", raising=False)
    return tmp_path


# ---------- scan ----------

def test_scan_finds_the_page_and_its_files(wiki):
    plan = removal.scan(STEM)
    assert plan.page_path == wiki / "wiki" / "compbio" / f"{STEM}.md"
    assert plan.page_type == "paper"
    assert wiki / "papers" / f"{STEM}.pdf" in plan.files


def test_scan_finds_backlinks_and_index(wiki):
    plan = removal.scan(STEM)
    assert [p.name for p, _ in plan.backlink_pages] == ["jones-2025-another.md"]
    assert plan.backlink_pages[0][1] == 1
    assert plan.index_bullet is True


def test_scan_reports_authored_citations_separately(wiki):
    plan = removal.scan(STEM)
    paths = {r.path.name for r in plan.prose_refs}
    assert "a-field-map.md" in paths
    assert all(r.page_type in {"synthesis", "concept"} for r in plan.prose_refs)
    # A synthesis page is never a back-link target.
    assert "a-field-map.md" not in {p.name for p, _ in plan.backlink_pages}


def test_scan_finds_the_concept_hub(wiki):
    plan = removal.scan(STEM)
    assert [p.name for p in plan.concept_hubs] == ["pangenome.md"]


def test_scan_writes_nothing(wiki):
    before = {p: p.stat().st_mtime_ns for p in (wiki / "wiki").rglob("*.md")}
    removal.scan(STEM)
    after = {p: p.stat().st_mtime_ns for p in (wiki / "wiki").rglob("*.md")}
    assert before == after


def test_scan_of_an_unknown_stem_is_empty(wiki):
    assert removal.scan("nobody-1999-nothing").exists is False


def test_prose_ref_pages_are_not_in_the_snapshot_set(wiki):
    """They are not written, so declaring them would claim an intent this
    command doesn't have."""
    plan = removal.scan(STEM)
    assert wiki / "wiki" / "synthesis" / "a-field-map.md" not in plan.touched_paths


# ---------- apply ----------

def test_apply_removes_page_pdf_and_backlink(wiki):
    removal.apply(removal.scan(STEM))
    assert not (wiki / "wiki" / "compbio" / f"{STEM}.md").exists()
    assert not (wiki / "papers" / f"{STEM}.pdf").exists()
    other = (wiki / "wiki" / "compbio" / "jones-2025-another.md").read_text(encoding="utf-8")
    assert STEM not in other
    assert "keep-2020-unrelated" in other, "unrelated bullets must survive"


def test_apply_removes_the_index_bullet_only(wiki):
    removal.apply(removal.scan(STEM))
    idx = (wiki / "wiki" / "index.md").read_text(encoding="utf-8")
    assert STEM not in idx
    assert "jones-2025-another" in idx


def test_apply_never_edits_a_synthesis_page(wiki):
    """The decided policy, asserted on bytes."""
    page = wiki / "wiki" / "synthesis" / "a-field-map.md"
    before = page.read_text(encoding="utf-8")
    removal.apply(removal.scan(STEM))
    assert page.read_text(encoding="utf-8") == before


def test_apply_strips_the_concept_spoke_but_not_the_definition(wiki):
    hub = wiki / "wiki" / "concepts" / "pangenome.md"
    removal.apply(removal.scan(STEM))
    text = hub.read_text(encoding="utf-8")
    assert STEM not in text, "the generated spoke registry is cleaned"
    assert "A pangenome is a thing." in text, "authored Definition survives"
    assert "other-2023-thing" in text, "the other spoke survives"


def test_apply_recomputes_concept_span(wiki):
    hub = wiki / "wiki" / "concepts" / "pangenome.md"
    removal.apply(removal.scan(STEM))
    assert "concept_span: 1" in hub.read_text(encoding="utf-8")


def test_keep_pdf_leaves_the_pdf(wiki):
    removal.apply(removal.scan(STEM), keep_pdf=True)
    assert (wiki / "papers" / f"{STEM}.pdf").exists()
    assert not (wiki / "wiki" / "compbio" / f"{STEM}.md").exists()


# ---------- CLI ----------

def test_dry_run_is_the_default_and_writes_nothing(wiki, capsys):
    before = {p: p.read_bytes() for p in (wiki / "wiki").rglob("*.md")}
    assert cli.main([STEM]) == 0
    assert {p: p.read_bytes() for p in (wiki / "wiki").rglob("*.md")} == before
    out = capsys.readouterr().out
    assert "dry run" in out
    assert (wiki / "papers" / f"{STEM}.pdf").exists()


def test_dry_run_names_the_authored_citations(wiki, capsys):
    cli.main([STEM])
    out = capsys.readouterr().out
    assert "authored citation" in out
    assert "a-field-map.md" in out
    assert "NOT edited" in out


def test_unknown_stem_exits_1(wiki, capsys):
    assert cli.main(["nobody-1999-nothing"]) == 1
    assert "nothing found" in capsys.readouterr().err


def test_apply_writes_a_log_entry(wiki):
    cli.main([STEM, "--apply"])
    log = (wiki / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "remove" in log and STEM in log


def test_apply_leaves_no_journal(wiki):
    cli.main([STEM, "--apply"])
    assert mut.pending_journals() == []


def test_json_mode_reports_the_plan(wiki, capsys):
    assert cli.main([STEM, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stem"] == STEM
    assert payload["applied"] is False
    assert payload["index_bullet"] is True
    assert any("a-field-map" in r["path"] for r in payload["prose_refs"])


# ---------- directory traces (.supp/, caches) ----------

def test_apply_removes_a_paper_with_supplementary_files(wiki):
    """Directories in the snapshot set used to kill the removal outright:
    `shutil.copy2` on `papers/{stem}.supp/` raised IsADirectoryError before
    anything was removed."""
    supp = wiki / "papers" / f"{STEM}.supp"
    supp.mkdir()
    (supp / "tableS1.csv").write_text("supp,data\n", encoding="utf-8")

    assert cli.main([STEM, "--apply"]) == 0
    assert not supp.exists()
    assert not (wiki / "wiki" / "compbio" / f"{STEM}.md").exists()
    assert mut.pending_journals() == []


def test_a_failure_mid_removal_restores_the_supp_directory(wiki, monkeypatch):
    supp = wiki / "papers" / f"{STEM}.supp"
    supp.mkdir()
    (supp / "tableS1.csv").write_text("supp,data\n", encoding="utf-8")

    monkeypatch.setattr(removal, "_strip_index_bullet",
                        lambda stem: (_ for _ in ()).throw(OSError("disk full")))

    assert cli.main([STEM, "--apply"]) == 2
    assert (supp / "tableS1.csv").read_text(encoding="utf-8") == "supp,data\n"
    assert (wiki / "papers" / f"{STEM}.pdf").exists()


# ---------- stem anchoring ----------

def test_a_stem_suffix_collision_survives_removal(wiki):
    """`garcia-smith-…` contains STEM (`smith-…`) as a suffix — the shape a
    hyphenated surname produces. The earlier substring/suffix regexes deleted
    the other paper's bullets and spokes along with the target's."""
    other = f"garcia-{STEM}"
    (wiki / "wiki" / "compbio" / f"{other}.md").write_text(
        "---\ntype: paper\n---\n\n# Garcia-Smith\n", encoding="utf-8")
    idx = wiki / "wiki" / "index.md"
    idx.write_text(idx.read_text(encoding="utf-8")
                   + f"- [[compbio/{other}]] — **Garcia-Smith 2024**: sibling.\n",
                   encoding="utf-8")
    hub = wiki / "wiki" / "concepts" / "pangenome.md"
    hub.write_text(hub.read_text(encoding="utf-8")
                   + f"- [[compbio/{other}]] — also builds one\n", encoding="utf-8")

    removal.apply(removal.scan(STEM))

    assert other in idx.read_text(encoding="utf-8"), \
        "the suffix-colliding paper keeps its index bullet"
    assert other in hub.read_text(encoding="utf-8"), \
        "the suffix-colliding paper keeps its concept spoke"


# ---------- commentary detection ----------

def test_commentary_is_flagged_only_via_primary_paper(wiki):
    """A commentary whose *body* mentions the stem is not orphaned by the
    removal; only one naming it in `primary_paper:` is. The old probe used
    DOTALL and matched to end-of-file."""
    (wiki / "wiki" / "compbio" / "marchal-2026-highlight.md").write_text(
        "---\ntype: commentary\n"
        'primary_paper: "[[compbio/keep-2020-unrelated]]"\n---\n\n'
        f"## Summary\nDiscusses [[compbio/{STEM}]] in passing.\n",
        encoding="utf-8")
    (wiki / "wiki" / "compbio" / "wu-2026-news-and-views.md").write_text(
        "---\ntype: commentary\nprimary_paper:\n"
        f'  - "[[compbio/{STEM}]]"\n'
        '  - "Wu et al. (2026) — companion; not in this wiki"\n---\n\n'
        "## Summary\nCovers the primary study.\n",
        encoding="utf-8")

    plan = removal.scan(STEM)
    names = {p.name for p in plan.commentary_pages}
    assert names == {"wu-2026-news-and-views.md"}


# ---------- rollback ----------

def test_a_failure_mid_removal_restores_everything(wiki, monkeypatch):
    """`remove` is the journal's second consumer; a partial removal is exactly
    the state it exists to prevent."""
    other = wiki / "wiki" / "compbio" / "jones-2025-another.md"
    page = wiki / "wiki" / "compbio" / f"{STEM}.md"
    idx = wiki / "wiki" / "index.md"
    before = (page.read_bytes(), other.read_bytes(), idx.read_bytes())

    monkeypatch.setattr(removal, "_strip_index_bullet",
                        lambda stem: (_ for _ in ()).throw(OSError("disk full")))

    assert cli.main([STEM, "--apply"]) == 2
    assert (page.read_bytes(), other.read_bytes(), idx.read_bytes()) == before
    assert (wiki / "papers" / f"{STEM}.pdf").exists()


# ---------- non-paper targets ----------
#
# `scan` resolves the target by filename stem and never branches on `type:`, so
# every page type is removable. These pin that as contract rather than accident:
# the command's prose, its `--help`, and every test above speak of papers, so a
# refactor that added a `type: paper` guard would otherwise pass green.

def _index_lines(wiki) -> list[str]:
    return (wiki / "wiki" / "index.md").read_text(encoding="utf-8").splitlines()


def test_a_synthesis_page_can_be_the_target(wiki):
    """The page goes, and so does its index bullet — the same treatment a
    paper gets, with the paper-shaped file probes finding nothing."""
    page = wiki / "wiki" / "synthesis" / "a-field-map.md"
    idx = wiki / "wiki" / "index.md"
    idx.write_text(idx.read_text(encoding="utf-8")
                   + "\n## synthesis\n\n- [[synthesis/a-field-map]] — a map.\n",
                   encoding="utf-8")

    plan = removal.scan("a-field-map")
    assert plan.page_path == page
    assert plan.page_type == "synthesis"
    assert plan.files == [], "no PDF, supp dir or cache exists for a synthesis page"
    assert plan.index_bullet is True

    removal.apply(plan)
    assert not page.exists()
    assert "a-field-map" not in idx.read_text(encoding="utf-8")


def test_removing_a_concept_hub_strips_its_reciprocal_backlinks(wiki):
    """Members carry a generated `[[concepts/<slug>]]` bullet. It is a
    back-link like any other, so it goes with the hub."""
    member = wiki / "wiki" / "compbio" / "jones-2025-another.md"
    member.write_text(member.read_text(encoding="utf-8")
                      + "- [[concepts/pangenome]] — instantiates the concept "
                        "(auto-added; concept-link)\n", encoding="utf-8")

    plan = removal.scan("pangenome")
    assert plan.page_type == "concept"
    assert [p.name for p, _ in plan.backlink_pages] == ["jones-2025-another.md"]

    removal.apply(plan)
    assert not (wiki / "wiki" / "concepts" / "pangenome.md").exists()
    text = member.read_text(encoding="utf-8")
    assert "concepts/pangenome" not in text
    assert STEM in text, "the member paper's other bullets are untouched"


def test_an_authored_page_citing_the_target_is_still_only_reported(wiki):
    """`AUTHORED_TYPES` guards *citing* pages, not the target — so removing one
    authored page never rewrites another."""
    citing = wiki / "wiki" / "synthesis" / "b-field-map.md"
    citing.write_text(
        "---\ntype: synthesis\ncategory: [compbio]\n---\n\n"
        "## Question\nWhat?\n\n## Short answer\n"
        "Argued at length in [[concepts/pangenome]].\n",
        encoding="utf-8")
    before = citing.read_bytes()

    plan = removal.scan("pangenome")
    assert [r.path.name for r in plan.prose_refs] == ["b-field-map.md"]

    removal.apply(plan)
    assert citing.read_bytes() == before


def test_a_reference_doc_target_takes_its_pdf(wiki):
    """A guidance/protocol/whitepaper page has a PDF like a paper does, and it
    is not in AUTHORED_TYPES, so it gets the full paper treatment."""
    (wiki / "wiki" / "references").mkdir()
    stem = "fda-2026-human-gene-therapy-products-incorporating"
    page = wiki / "wiki" / "references" / f"{stem}.md"
    page.write_text("---\ntype: guidance\nissuer: FDA\n---\n\n## Summary\nbody\n",
                    encoding="utf-8")
    pdf = wiki / "papers" / f"{stem}.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    plan = removal.scan(stem)
    assert plan.page_type == "guidance"
    assert pdf in plan.files

    removal.apply(plan)
    assert not page.exists() and not pdf.exists()


def test_the_wiki_root_bookkeeping_pages_are_not_targets(wiki):
    """`_iter_pages` excludes them by name, so no stem can resolve to one."""
    for stem in ("index", "log"):
        assert removal.scan(stem).page_path is None
    assert (wiki / "wiki" / "index.md").exists()
    assert (wiki / "wiki" / "log.md").exists()


def test_the_summary_reports_the_index_bullet(wiki, capsys):
    """The plan promises `index.md  1 bullet`; the result has to confirm it."""
    assert cli.main([STEM, "--apply"]) == 0
    assert "1 index.md bullet" in capsys.readouterr().out


def test_keep_pdf_lands_the_stem_in_lint_orphan_pdfs(wiki):
    """The documented hand-off: `--keep-pdf` deliberately leaves a PDF with no
    page, and `lint`'s `orphan_pdfs` is where it then shows up as a re-ingest
    queue. Pinned end-to-end because the two features only make sense together
    — before the check existed the kept PDF was simply forgotten."""
    from researchwiki.wiki import find_orphan_pdfs

    assert find_orphan_pdfs() == []

    removal.apply(removal.scan(STEM), keep_pdf=True)

    assert (wiki / "papers" / f"{STEM}.pdf").exists()
    assert find_orphan_pdfs() == [STEM]
