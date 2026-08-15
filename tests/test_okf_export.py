"""`researchwiki export --format okf` — emitting an OKF v0.2 bundle.

The load-bearing assertions here are the ones that would let a *plausible* bundle
ship a false claim or a broken reference:

  - **Scope.** OKF carries every page type; the bibliography carries only
    published ones. Both are pinned, because "aligning" them is the tempting
    wrong fix.
  - **Conformance.** §11's three criteria, asserted over a generated bundle rather
    than trusted from the mapping code.
  - **`verified` is not invented.** A synthesis page has no persisted gate run, so
    it must carry no trust claim at all. Emitting one would be exactly the
    citation-shaped falsehood this repo exists to avoid.
  - **Reserved names.** `index.md`/`log.md` may not be concept documents (§3.1), so
    the wiki's own bookkeeping pages must not land as concepts.

Hermetic: tmp wiki, no DB, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from researchwiki import okfexport, paths


def _page(root: Path, key: str, body: str, **fm) -> Path:
    p = root / "wiki" / f"{key}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    front = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True)
    p.write_text(f"---\n{front}---\n\n{body}\n", encoding="utf-8")
    return p


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    # No claim DB in a tmp wiki -> `_graded_at` degrades to {} and no page can
    # claim a verified tier. Patched explicitly so the test does not depend on
    # whatever state.db the developer's machine happens to have.
    monkeypatch.setattr(okfexport, "_graded_at", lambda: {})

    _page(tmp_path, "cgt/smith-2024-a-paper-about-things",
          "## Summary\nSee [[cgt/jones-2025-another-paper]] and [[concepts/pangenome]].",
          type="paper", title="A paper about things", year=2024,
          doi="10.1038/example", hook="Does a thing with [[cgt/jones-2025-another-paper]].",
          keywords=["alpha", "beta"], author_model="claude-opus-4-7",
          ingested_at="2026-06-10T20:01:57", pdf_path="[[smith-2024-a-paper-about-things.pdf]]")
    _page(tmp_path, "cgt/jones-2025-another-paper", "## Summary\nBare link [[smith-2024-a-paper-about-things]].",
          type="paper", title="Another paper", year=2025, hook="Other thing.")
    _page(tmp_path, "concepts/pangenome",
          "## Definition\nA thing.\n\n## How it appears across the corpus\n"
          "- [[cgt/smith-2024-a-paper-about-things]] — builds one",
          type="concept", title="Pangenome", hook="A hub.",
          referenced_papers=["[[cgt/smith-2024-a-paper-about-things]]"], tags=["graphs"])
    _page(tmp_path, "synthesis/field-map",
          "## Question\nWhat?\n\n## Short answer\n"
          "Three assemblers agree[^smith] and one uses "
          "[[cgt/smith-2024-a-paper-about-things#kc-9f3a2b1c]].\n\n"
          "## References\n\n[^smith]: [[cgt/smith-2024-a-paper-about-things]] — Smith 2024\n",
          type="synthesis", title="A field map", hook="Maps a field.",
          category=["cgt"], tags=["synthesis"])
    _page(tmp_path, "ideas/a-design",
          "## Verdict\nstrong\n\n## Background\nGrounded[^smith].\n\n"
          "## References\n\n[^smith]: [[cgt/smith-2024-a-paper-about-things]]\n",
          type="idea", title="A design", hook="Proposes a thing.", status="scoping")
    # Bookkeeping pages that must NOT become concepts (§3.1 reserved names).
    (tmp_path / "wiki" / "index.md").write_text(
        "---\ntype: meta\n---\n\n## cgt\n\n- [[cgt/smith-2024-a-paper-about-things]] — x\n",
        encoding="utf-8")
    (tmp_path / "wiki" / "log.md").write_text(
        "# log\n\n## [2026-06-10] ingest | Smith 2024 — A paper about things\n"
        "Category: cgt.\n\n## [2026-06-09] query | Something → wiki/synthesis/field-map.md\n",
        encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------- scope

def test_okf_carries_every_page_type_unlike_the_bibliography(wiki):
    files, report = okfexport.collect_bundle()
    assert set(report.by_type) == {"Paper", "Concept", "Synthesis", "Idea"}

    # The contrast that must not be "fixed": the bibliography path excludes the
    # user's own analysis because a BibTeX entry would assert a publication.
    from researchwiki.refexport import EXPORTABLE_TYPES
    for t in ("synthesis", "idea", "concept"):
        assert t not in EXPORTABLE_TYPES
    assert "Synthesis" in report.by_type and "Idea" in report.by_type


def test_reserved_names_are_regenerated_not_carried_as_concepts(wiki):
    files, _ = okfexport.collect_bundle()
    assert "index.md" in files and "log.md" in files
    # the wiki's own bookkeeping pages must not appear as concept documents
    assert "wiki/index.md" not in files
    assert "wiki/log.md" not in files
    assert not any(k.endswith("/index.md") or k.endswith("/log.md") for k in files)


def test_abstract_concepts_carry_no_resource(wiki):
    files, _ = okfexport.collect_bundle()
    paper = yaml.safe_load(files["cgt/smith-2024-a-paper-about-things.md"].split("---\n")[1])
    synth = yaml.safe_load(files["synthesis/field-map.md"].split("---\n")[1])
    assert paper["resource"] == "https://doi.org/10.1038/example"
    assert "resource" not in synth      # §4.4 — absent, not empty


# ---------------------------------------------------------------- conformance

def test_generated_bundle_meets_okf_conformance(wiki, tmp_path):
    """§11's three criteria, over the real emitted bytes."""
    files, _ = okfexport.collect_bundle()
    for rel, text in files.items():
        if rel in ("index.md", "log.md"):
            continue
        assert text.startswith("---\n"), rel
        end = text.find("\n---\n", 4)
        assert end > 0, rel
        fm = yaml.safe_load(text[4:end])
        assert isinstance(fm, dict), rel
        assert isinstance(fm.get("type"), str) and fm["type"].strip(), rel

    idx = files["index.md"]
    assert "okf_version" in idx[:200]
    assert re.search(r"^# \w", idx, re.M)
    assert re.search(r"^\* \[.+\]\(/.+\.md\)", idx, re.M)
    log = files["log.md"]
    assert log.startswith("# Directory Update Log")
    assert re.search(r"^## \d{4}-\d{2}-\d{2}$", log, re.M)


def test_no_wikilinks_survive_anywhere(wiki):
    files, _ = okfexport.collect_bundle()
    for rel, text in files.items():
        assert "[[" not in text, f"{rel} still carries wikilink syntax"


# ---------------------------------------------------------------- links

def test_links_become_bundle_relative_and_bare_stems_resolve(wiki):
    files, report = okfexport.collect_bundle()
    smith = files["cgt/smith-2024-a-paper-about-things.md"]
    assert "[Another paper](/cgt/jones-2025-another-paper.md)" in smith
    # Obsidian resolves a bare stem; so must the bundle, or the edge vanishes.
    jones = files["cgt/jones-2025-another-paper.md"]
    assert "(/cgt/smith-2024-a-paper-about-things.md)" in jones
    assert report.links_rewritten > 0


def test_claim_anchors_survive_as_markdown_fragments(wiki):
    files, _ = okfexport.collect_bundle()
    synth = files["synthesis/field-map.md"]
    assert "(/cgt/smith-2024-a-paper-about-things.md#kc-9f3a2b1c)" in synth


def test_unresolved_link_keeps_its_text_and_is_reported(wiki, tmp_path):
    _page(tmp_path, "cgt/dangling-2020-a-paper",
          "## Summary\nPoints at [[cgt/never-written-2019-nothing]].",
          type="paper", title="Dangling", hook="h")
    files, report = okfexport.collect_bundle()
    body = files["cgt/dangling-2020-a-paper.md"]
    assert "](/cgt/never-written" not in body          # no path to a missing file
    assert "never written 2019 nothing" in body        # §6.1 — information survives
    assert any(u["target"] == "cgt/never-written-2019-nothing"
               for u in report.links_unresolved)


def test_description_is_plain_text_not_link_syntax(wiki):
    files, _ = okfexport.collect_bundle()
    fm = yaml.safe_load(files["cgt/smith-2024-a-paper-about-things.md"].split("---\n")[1])
    # `hook:` held a wikilink; a one-line description must not carry link syntax
    assert "[[" not in fm["description"] and "](" not in fm["description"]
    assert "Another paper" in fm["description"]


# ---------------------------------------------------------------- provenance

def test_sources_ids_match_the_body_footnote_labels(wiki):
    """OKF resolves per-claim attribution through `sources[].id` (§5.1), and the
    wiki's footnote labels already are stable keys — so they must carry over
    unchanged or every footnote in the bundle dangles."""
    files, report = okfexport.collect_bundle()
    text = files["synthesis/field-map.md"]
    fm = yaml.safe_load(text.split("---\n")[1])
    ids = {s["id"] for s in fm["sources"]}
    assert ids == {"smith"}
    assert "[^smith]" in text                       # the citation still in the body
    assert fm["sources"][0]["resource"] == "/cgt/smith-2024-a-paper-about-things.md"
    assert report.sources_emitted >= 1


def test_concept_spokes_become_sources(wiki):
    files, _ = okfexport.collect_bundle()
    fm = yaml.safe_load(files["concepts/pangenome.md"].split("---\n")[1])
    assert [s["resource"] for s in fm["sources"]] == [
        "/cgt/smith-2024-a-paper-about-things.md"]


def test_generated_is_omitted_rather_than_invented(wiki):
    files, report = okfexport.collect_bundle()
    withmodel = yaml.safe_load(
        files["cgt/smith-2024-a-paper-about-things.md"].split("---\n")[1])
    assert withmodel["generated"] == {
        "by": "researchwiki/claude-opus-4-7", "at": "2026-06-10T20:01:57"}
    # `by` is REQUIRED inside `generated` (§5.2); with no recorded author the whole
    # block is dropped instead of naming a process we cannot vouch for.
    nomodel = yaml.safe_load(files["cgt/jones-2025-another-paper.md"].split("---\n")[1])
    assert "generated" not in nomodel


def test_verified_is_never_claimed_without_a_persisted_gate_run(wiki):
    """The whole point. `check-grounding` / `grade synthesis` persist nothing, so a
    synthesis page cannot honestly carry a trust tier."""
    files, report = okfexport.collect_bundle()
    for key in ("synthesis/field-map.md", "ideas/a-design.md", "concepts/pangenome.md"):
        fm = yaml.safe_load(files[key].split("---\n")[1])
        assert "verified" not in fm, key
    assert set(report.verified_absent_no_gate_record) == {
        "synthesis/field-map", "ideas/a-design", "concepts/pangenome"}
    assert report.verified_emitted == 0


def test_verified_is_emitted_for_graded_papers(wiki, monkeypatch):
    monkeypatch.setattr(okfexport, "_graded_at",
                        lambda: {"smith-2024-a-paper-about-things": 1_780_000_000})
    files, report = okfexport.collect_bundle()
    fm = yaml.safe_load(files["cgt/smith-2024-a-paper-about-things.md"].split("---\n")[1])
    assert fm["verified"]["by"] == "process:researchwiki-grade-paper"
    assert fm["verified"]["at"].endswith("Z")
    # machine-confirmed, not human-reviewed (§5.3) — no `human:` prefix
    assert not fm["verified"]["by"].startswith("human:")
    assert report.verified_emitted == 1


def test_idea_status_maps_and_keeps_the_native_value(wiki):
    files, _ = okfexport.collect_bundle()
    fm = yaml.safe_load(files["ideas/a-design.md"].split("---\n")[1])
    assert fm["status"] == "draft"                       # scoping -> draft (§5.4)
    assert fm["x_researchwiki_status"] == "scoping"      # nothing lost


def test_tags_union_keywords_so_neither_field_is_lost(wiki):
    files, _ = okfexport.collect_bundle()
    paper = yaml.safe_load(files["cgt/smith-2024-a-paper-about-things.md"].split("---\n")[1])
    concept = yaml.safe_load(files["concepts/pangenome.md"].split("---\n")[1])
    assert paper["tags"] == ["alpha", "beta"]     # from `keywords:`
    assert concept["tags"] == ["graphs"]          # from `tags:`


def test_vault_plumbing_is_dropped_not_exported(wiki):
    files, _ = okfexport.collect_bundle()
    fm = yaml.safe_load(files["cgt/smith-2024-a-paper-about-things.md"].split("---\n")[1])
    # `pdf_path` names a file that isn't in the bundle; exporting it would be a
    # path-valued field pointing at nothing.
    assert not any(k.endswith("pdf_path") for k in fm)


def test_mapped_keys_are_not_also_duplicated_as_extension_keys(wiki, tmp_path):
    """`source_url` → `resource` and `generated_at` → `generated.at` are mapped;
    they must not additionally leak through the `x_researchwiki_` passthrough
    the way unmapped keys do."""
    _page(tmp_path, "references/acme-2026-a-whitepaper",
          "## Summary\nA whitepaper.",
          type="whitepaper", title="Acme whitepaper", hook="A whitepaper.",
          source_url="https://acme.example/wp.pdf",
          author_model="claude-opus-4-7", generated_at="2026-06-11")
    files, _ = okfexport.collect_bundle()
    fm = yaml.safe_load(files["references/acme-2026-a-whitepaper.md"].split("---\n")[1])
    assert fm["resource"] == "https://acme.example/wp.pdf"
    assert fm["generated"]["at"] == "2026-06-11"
    assert "x_researchwiki_source_url" not in fm
    assert "x_researchwiki_generated_at" not in fm


# ---------------------------------------------------------------- log

def test_log_translates_to_okf_shape_newest_first(wiki):
    files, _ = okfexport.collect_bundle()
    log = files["log.md"]
    assert log.index("## 2026-06-10") < log.index("## 2026-06-09")
    assert "* **Ingest**: Smith 2024 — A paper about things" in log
    assert "* **Query**: Something" in log


def test_log_can_be_excluded(wiki):
    files, _ = okfexport.collect_bundle(include_log=False)
    assert "log.md" not in files


# ---------------------------------------------------------------- write

def test_write_bundle_reports_stale_files_without_deleting_them(wiki, tmp_path):
    out = tmp_path / "bundle"
    files, _ = okfexport.collect_bundle()
    okfexport.write_bundle(files, out)
    leftover = out / "cgt" / "removed-2019-a-paper.md"
    leftover.write_text("---\ntype: Paper\n---\n\nold\n", encoding="utf-8")

    stale = okfexport.write_bundle(files, out)
    assert "cgt/removed-2019-a-paper.md" in stale
    assert leftover.exists(), "a file the user pointed us at must not be deleted"


def test_bundle_detection_keys_on_okf_version(wiki, tmp_path):
    out = tmp_path / "b"
    out.mkdir()
    assert not okfexport.looks_like_okf_bundle(out)
    files, _ = okfexport.collect_bundle()
    okfexport.write_bundle(files, out)
    assert okfexport.looks_like_okf_bundle(out)


def test_export_is_deterministic(wiki):
    a, _ = okfexport.collect_bundle()
    b, _ = okfexport.collect_bundle()
    assert a == b


# ---------------------------------------------------------------- CLI

def test_cli_requires_out_for_okf(wiki, monkeypatch, capsys):
    from researchwiki.tasks import export as cli
    monkeypatch.chdir(wiki)
    assert cli.main(["--format", "okf"]) == 1
    assert "needs --out" in capsys.readouterr().err


def test_cli_refuses_a_non_bundle_directory(wiki, monkeypatch, tmp_path):
    from researchwiki.errors import EnvironmentFailure
    from researchwiki.tasks import export as cli
    monkeypatch.chdir(wiki)
    victim = tmp_path / "mine"
    victim.mkdir()
    (victim / "important.txt").write_text("do not clobber", encoding="utf-8")
    with pytest.raises(EnvironmentFailure, match="refusing to write"):
        cli.main(["--format", "okf", "--out", str(victim)])
    assert (victim / "important.txt").read_text(encoding="utf-8") == "do not clobber"


def test_cli_writes_a_bundle_and_reports(wiki, monkeypatch, tmp_path, capsys):
    from researchwiki.tasks import export as cli
    monkeypatch.chdir(wiki)
    out = tmp_path / "okf"
    assert cli.main(["--format", "okf", "--out", str(out)]) == 0
    assert (out / "index.md").is_file()
    assert (out / "cgt" / "smith-2024-a-paper-about-things.md").is_file()
    err = capsys.readouterr().err
    assert "okf:" in err and "concepts" in err
    # the trust gap must be stated, or "unverified" reads as "ungraded"
    assert "no `verified`" in err


def test_cli_json_mode_emits_the_okf_report(wiki, monkeypatch, tmp_path, capsys):
    import json
    from researchwiki.tasks import export as cli
    monkeypatch.chdir(wiki)
    assert cli.main(["--format", "okf", "--out", str(tmp_path / "okf"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "okf"
    assert payload["okf_version"] == okfexport.OKF_VERSION
    assert payload["concepts"] == 5
    assert "stale_files" in payload


def test_a_reserved_stem_in_a_category_dir_is_skipped_and_reported(wiki, tmp_path):
    """§3.1 reserves `index.md`/`log.md` at *every* level, not just the root.

    `wiki/cgt/index.md` is a legal wiki page, but emitting it as `cgt/index.md`
    would give the bundle a second directory listing and a consumer would read it
    as one. Skipped, and named in the report — the fix is to rename the page, and
    dropping it silently would leave the omission invisible.
    """
    _page(tmp_path, "cgt/index", "## Summary\nnot really a listing", type="paper",
          title="Sneaky", hook="h")
    files, report = okfexport.collect_bundle()
    assert "cgt/index.md" not in files
    assert any(s["page"] == "cgt/index" and "reserved" in s["reason"]
               for s in report.skipped)


def test_wiki_root_bookkeeping_is_skipped_quietly(wiki):
    """The root `index.md`/`log.md` are *expected* to be excluded — they get
    regenerated — so they must not clutter the report the way a rename-me page
    does."""
    _, report = okfexport.collect_bundle()
    assert not any(s["page"].startswith("wiki/") for s in report.skipped)
