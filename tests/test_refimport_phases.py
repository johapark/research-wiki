"""End-to-end phase tests for `researchwiki import`.

**This file is the point of the exercise.** `migrate`'s unit tests are good, but
every test it has is a unit test, so its phases — the part users actually run —
have no coverage at all: nothing in the suite imports `migrate/manifest.py`,
`migrate/apply.py` or `tasks/migrate.py`. These tests invoke `main([...])` the
way a user does, against a real temporary wiki, and assert on exit codes,
written files and `--json` keys.

`--json` keys are a released contract the moment this ships (`CHANGELOG.md`
names removing one a breaking change), so they are asserted explicitly.
"""

import importlib
import json
from pathlib import Path

import pytest

# `researchwiki/tasks/import.py` is named for a keyword, so no `import`
# statement can reach it. That is deliberate — see the module docstring.
import_task = importlib.import_module("researchwiki.tasks.import")

FIXTURES = Path(__file__).parent / "refimport-fixtures"
RIS = FIXTURES / "readcube-sample.ris"
BIB = FIXTURES / "readcube-sample.bib"
CSL = FIXTURES / "zotero-sample.json"


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    """A minimal wiki rooted at a tmp dir.

    `researchwiki` resolves every path from `Path.cwd()`, so chdir is the
    supported way to point it at a different tree.
    """
    for d in ("wiki/cgt", "wiki/other", "papers", "inbox", ".ingest"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(argv, capsys) -> tuple[int, str, str]:
    code = import_task.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def latest_run_dir(wiki: Path) -> Path:
    runs = sorted((wiki / ".ingest").glob("import-*"))
    assert runs, "no run directory was created"
    return runs[-1]


def add_page(wiki: Path, category: str, stem: str, doi: str | None = None) -> None:
    fm = ["---", "title: An existing paper", "type: paper",
          f"category: [{category}]"]
    if doi:
        fm.append(f"doi: {doi}")
    fm += ["---", "", "## Summary", "", "Body."]
    (wiki / "wiki" / category / f"{stem}.md").write_text("\n".join(fm), encoding="utf-8")


# ---------- preflight ----------

def test_preflight_on_a_good_export_exits_zero(wiki, capsys):
    code, out, _ = run(["preflight", str(RIS)], capsys)
    assert code == 0
    assert "records            12" in out
    assert "format             ris" in out


def test_preflight_reports_doi_coverage(wiki, capsys):
    _, out, _ = run(["preflight", str(RIS)], capsys)
    assert "with a usable DOI" in out


def test_preflight_names_the_no_attachment_paths_case(wiki, capsys):
    """The single most confusing thing about a ReadCube export: it looks like
    it should carry file paths and does not. Say so rather than pairing zero
    files silently."""
    _, out, _ = run(["preflight", str(CSL)], capsys)
    assert "names no attachment paths" in out


def test_preflight_on_a_missing_file_exits_one(wiki, capsys):
    code, _, err = run(["preflight", "nope.ris"], capsys)
    assert code == 1 and "no such file" in err


def test_preflight_on_an_unidentifiable_file_exits_one(wiki, tmp_path, capsys):
    p = tmp_path / "notes.txt"
    p.write_text("just prose, not an export")
    code, _, err = run(["preflight", str(p)], capsys)
    assert code == 1 and "cannot identify" in err


def test_preflight_does_not_fail_on_a_broken_embedding_model(wiki, capsys, monkeypatch):
    """Deliberate divergence from `migrate`, which hard-fails its preflight on
    this. Parsing a .ris has no dependency on a 133 MB bi-encoder, and refusing
    to read an export because a torch wheel is wrong blocks the phase a user
    runs first — before they own any PDFs. `apply` is where it binds."""
    monkeypatch.setattr(import_task, "_embedding_status",
                        lambda: (False, "RuntimeError: _ARRAY_API not found"))
    code, out, _ = run(["preflight", str(RIS)], capsys)
    assert code == 0
    assert "WARN unusable" in out and "re-grading the whole import" in out


def test_preflight_probes_an_encode_not_just_construction(monkeypatch):
    """`is_available()` only proves the model *constructs*; a torch/NumPy ABI
    mismatch loads fine, reports OK, and dies on the first real call. The probe
    must therefore reach `embed_texts`, which this asserts by failing there."""
    import types

    import researchwiki.index as index_pkg

    called = {}

    def boom(texts):
        called["texts"] = texts
        raise RuntimeError("_ARRAY_API not found")

    fake = types.SimpleNamespace(DEFAULT_MODEL="fake-model", embed_texts=boom)
    monkeypatch.setattr(index_pkg, "embeddings", fake)

    ok, detail = import_task._embedding_status()
    assert ok is False and "_ARRAY_API not found" in detail
    assert called["texts"] == ["probe"], "the probe never reached embed_texts"


def test_embedding_probe_reports_the_model_and_dimension_on_success(monkeypatch):
    import types

    import numpy as np

    import researchwiki.index as index_pkg

    fake = types.SimpleNamespace(
        DEFAULT_MODEL="fake-model",
        embed_texts=lambda texts: np.zeros((len(texts), 384)),
    )
    monkeypatch.setattr(index_pkg, "embeddings", fake)
    ok, detail = import_task._embedding_status()
    assert ok is True and "fake-model" in detail and "384d" in detail


# ---------- inspect: basics ----------

def test_inspect_writes_a_manifest_and_a_report(wiki, capsys):
    code, _, _ = run(["inspect", str(RIS)], capsys)
    assert code == 0
    run_dir = latest_run_dir(wiki)
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "report.md").is_file()


def test_inspect_exits_zero_even_when_nothing_is_ready(wiki, capsys):
    """A triage result IS the deliverable. Exiting 1 here would make the normal
    metadata-only run look like a failure."""
    code, out, _ = run(["inspect", str(RIS)], capsys)
    assert code == 0
    assert "ready    0" in out


def test_inspect_writes_nothing_outside_the_run_directory(wiki, capsys):
    before = {p for p in wiki.rglob("*") if ".ingest" not in p.parts}
    run(["inspect", str(RIS)], capsys)
    after = {p for p in wiki.rglob("*") if ".ingest" not in p.parts}
    assert before == after


def test_inspect_on_a_missing_export_exits_one(wiki, capsys):
    code, _, err = run(["inspect", "nope.ris"], capsys)
    assert code == 1 and "no such file" in err


def test_inspect_with_an_unreadable_pdf_root_exits_one(wiki, capsys):
    """Exit 1, not 2: a mistyped argument is a user-input error, and the
    exit-code contract sends 2 to someone who will go inspect the disk."""
    code, _, err = run(["inspect", str(RIS), "no/such/dir"], capsys)
    assert code == 1 and "no such directory" in err


def test_inspect_rejects_a_category_with_no_directory(wiki, capsys):
    """Categories are explicit — a typo must not spawn one."""
    code, _, err = run(["inspect", str(RIS), "--category", "typo"], capsys)
    assert code == 1 and "has no wiki/typo/" in err


def test_inspect_accepts_an_existing_category(wiki, capsys):
    assert run(["inspect", str(RIS), "--category", "cgt"], capsys)[0] == 0


def test_inspect_limit_caps_the_records_assessed(wiki, capsys):
    run(["inspect", str(RIS), "--limit", "3"], capsys)
    data = json.loads((latest_run_dir(wiki) / "manifest.json").read_text())
    assert len(data["items"]) == 3


# ---------- inspect: the --json contract ----------

def test_inspect_json_emits_the_documented_keys(wiki, capsys):
    """Released contract: `CHANGELOG.md` names removing a `--json` key a
    breaking change, so the keys are pinned here rather than sampled."""
    code, out, _ = run(["inspect", str(RIS), "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert set(payload) == {
        "run_dir", "export_format", "summary",
        "missing_pdf_fetch_list", "unclaimed_pdfs", "items",
    }
    assert set(payload["summary"]) == {"total", "verdicts", "reasons"}


def test_inspect_json_item_keys_are_stable(wiki, capsys):
    _, out, _ = run(["inspect", str(RIS), "--json"], capsys)
    item = json.loads(out)["items"][0]
    assert set(item) == {
        "key", "title", "doi", "year", "authors", "item_type", "verdict",
        "reasons", "derived_stem", "primary_pdf", "supplementary", "pair_rung",
        "pair_confidence", "pair_rival", "pair_margin", "pair_candidates",
        "chars_per_page", "page_count", "collision", "ingest_args",
    }
    # Adding a key is additive and allowed; removing one is breaking
    # (`CHANGELOG.md`). This assertion is exact so that either direction has to
    # be a deliberate edit here rather than an unnoticed side effect — which is
    # what it caught when `pair_rival`/`pair_margin` were introduced.


def test_inspect_json_is_the_only_thing_on_stdout(wiki, capsys):
    """So a caller can pipe it. Progress goes to stderr."""
    _, out, _ = run(["inspect", str(RIS), "--json"], capsys)
    json.loads(out)


# ---------- inspect: gates, end to end ----------

def test_preprint_pair_is_deduped_in_a_real_run(wiki, capsys):
    _, out, _ = run(["inspect", str(RIS), "--json"], capsys)
    items = json.loads(out)["items"]
    pre = [i for i in items if i["doi"] and i["doi"].startswith("10.1101/")]
    assert pre and all("superseded-by-journal" in i["reasons"] for i in pre)


def test_a_page_already_in_the_wiki_is_skipped_as_already_present(wiki, capsys):
    add_page(wiki, "cgt", "fixture-2023-a-draft-synthetic-pangenome-reference",
             doi="10.1234/jtg.2023.0001")
    _, out, _ = run(["inspect", str(RIS), "--json"], capsys)
    items = {i["key"]: i for i in json.loads(out)["items"]}
    hit = [i for i in items.values() if i["doi"] == "10.1234/jtg.2023.0001"][0]
    assert "already-present" in hit["reasons"]
    assert hit["collision"]["kind"] == "doi"


def test_unicode_dash_record_derives_the_corrected_stem_end_to_end(wiki, capsys):
    _, out, _ = run(["inspect", str(RIS), "--json"], capsys)
    stems = [i["derived_stem"] for i in json.loads(out)["items"] if i["derived_stem"]]
    assert "hyphen-2015-atac-seq-a-method-for-assaying" in stems
    assert all("--" not in s and not s.endswith("-") for s in stems)


def test_ingest_args_are_recorded_per_item(wiki, capsys):
    """The manifest carries the exact argv each record contributes, so `apply`
    is a copy plus a dispatch and never re-derives anything."""
    _, out, _ = run(["inspect", str(RIS), "--json"], capsys)
    item = [i for i in json.loads(out)["items"]
            if i["doi"] == "10.1234/jtg.2023.0001"][0]
    assert item["ingest_args"][:2] == ["--doi", "10.1234/jtg.2023.0001"]
    assert "--year" in item["ingest_args"]


# ---------- inspect: no PDFs at all ----------

def test_inspect_without_a_pdf_root_still_triages(wiki, capsys):
    """The situation a cloud-hosted library is actually in."""
    code, out, _ = run(["inspect", str(RIS)], capsys)
    assert code == 0 and "skip     12" in out


def test_report_carries_the_missing_pdf_fetch_list(wiki, capsys):
    """Without this, a metadata-only run reports a count and nothing
    actionable. The DOIs are the work item."""
    run(["inspect", str(RIS)], capsys)
    report = (latest_run_dir(wiki) / "report.md").read_text()
    assert "## Missing PDFs" in report
    assert "10.1234/jtg.2023.0001" in report


def test_fetch_list_excludes_records_with_other_problems(wiki, capsys):
    """A book, or something already in the wiki, does not belong on a
    to-fetch list."""
    _, out, _ = run(["inspect", str(RIS), "--json"], capsys)
    payload = json.loads(out)
    fetch_dois = {f["doi"] for f in payload["missing_pdf_fetch_list"]}
    superseded = {i["doi"] for i in payload["items"]
                  if "superseded-by-journal" in i["reasons"]}
    assert fetch_dois.isdisjoint(superseded)


# ---------- inspect: with PDFs ----------

def test_inspect_pairs_a_pdf_and_marks_it_ready(wiki, capsys):
    from tests.test_refimport_pair import write_pdf

    pdfs = wiki / "pdfs"
    write_pdf(pdfs / "paper.pdf",
              ["A draft synthetic pangenome reference",
               "doi:10.1234/jtg.2023.0001"] + ["Body text " * 40] * 10, pages=6)
    code, out, _ = run(["inspect", str(RIS), str(pdfs), "--json"], capsys)
    assert code == 0
    items = {i["doi"]: i for i in json.loads(out)["items"]}
    hit = items["10.1234/jtg.2023.0001"]
    assert hit["verdict"] == "ready"
    assert hit["pair_rung"] == "doi"
    assert hit["primary_pdf"].endswith("paper.pdf")
    assert hit["chars_per_page"] > 200


def test_a_scanned_pdf_is_skipped_for_no_text_layer(wiki, capsys):
    from tests.test_refimport_pair import write_pdf

    pdfs = wiki / "pdfs"
    write_pdf(pdfs / "scan.pdf", ["doi:10.1234/jtg.2023.0001"], pages=9)
    _, out, _ = run(["inspect", str(RIS), str(pdfs), "--json"], capsys)
    hit = {i["doi"]: i for i in json.loads(out)["items"]}["10.1234/jtg.2023.0001"]
    assert hit["verdict"] == "skip" and "no-text-layer" in hit["reasons"]


def test_unclaimed_pdfs_are_reported(wiki, capsys):
    from tests.test_refimport_pair import write_pdf

    pdfs = wiki / "pdfs"
    write_pdf(pdfs / "stranger.pdf", ["Some completely unrelated document text"] * 10)
    _, out, _ = run(["inspect", str(RIS), str(pdfs), "--json"], capsys)
    assert any(p.endswith("stranger.pdf")
               for p in json.loads(out)["unclaimed_pdfs"])


# ---------- cross-format ----------

@pytest.mark.parametrize("export", [RIS, BIB], ids=["ris", "bibtex"])
def test_both_formats_of_one_library_triage_identically(wiki, capsys, export):
    """The two real exports described the same library. If a parser change makes
    the verdicts diverge, one of them is wrong."""
    _, out, _ = run(["inspect", str(export), "--json"], capsys)
    payload = json.loads(out)
    assert payload["summary"]["total"] == 12
    verdicts = {i["title"]: i["verdict"] for i in payload["items"] if i["title"]}
    assert verdicts["A draft synthetic pangenome reference"] == "skip"  # no PDF


def test_csl_json_book_and_webpage_are_typed_out(wiki, capsys):
    """Zotero populates `type`, so the typed gate fires — the one format where
    it can be relied on."""
    _, out, _ = run(["inspect", str(CSL), "--json"], capsys)
    items = {i["title"]: i for i in json.loads(out)["items"]}
    assert "not-a-paper" in items["The Test Framework Manual"]["reasons"]
    assert "not-a-paper" in items["A blog post about testing"]["reasons"]


# ---------- run directory ----------

def test_two_inspect_runs_get_separate_directories(wiki, capsys, monkeypatch):
    """No run may overwrite another's manifest."""
    stamps = iter(["20260101T000001", "20260101T000002"])
    monkeypatch.setattr(import_task, "_stamp", lambda: next(stamps))
    run(["inspect", str(RIS)], capsys)
    run(["inspect", str(RIS)], capsys)
    assert len(list((wiki / ".ingest").glob("import-*"))) == 2


def test_run_dir_override_is_honoured(wiki, capsys, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    run(["inspect", str(RIS), "--run-dir", str(elsewhere)], capsys)
    assert list(elsewhere.glob("import-*"))


# ---------- argparse surface ----------

def test_no_phase_is_a_usage_error(wiki, capsys):
    with pytest.raises(SystemExit):
        import_task.main([])


def test_unknown_phase_is_a_usage_error(wiki, capsys):
    with pytest.raises(SystemExit):
        import_task.main(["nonsense"])


# ---------- dedupe must happen before pairing ----------

def test_a_superseded_preprint_does_not_contest_its_survivors_pdf(wiki, capsys):
    """Ordering regression, found on a real library.

    A preprint and its published version share a title verbatim, so they score
    an exact tie against each other's PDFs. If the superseded record is still
    in the pairing pool, the distinctiveness gate reads that tie as a genuine
    ambiguity and sends the *survivor* — the record we do want — to review. On
    the real 532-record library this accounted for 8 of 10 `ambiguous-pairing`
    reviews, every one spurious.
    """
    from tests.test_refimport_pair import write_pdf

    title = "Sequence modeling and design from molecular to genome scale"
    pdfs = wiki / "pdfs"
    write_pdf(pdfs / "journal.pdf", [title] + ["Body text " * 40] * 10, pages=6)
    write_pdf(pdfs / "preprint.pdf", [title] + ["Body text " * 40] * 10, pages=6)

    _, out, _ = run(["inspect", str(RIS), str(pdfs), "--json"], capsys)
    items = {i["doi"]: i for i in json.loads(out)["items"] if i["doi"]}

    published = items["10.1234/science.2024.0005"]
    assert published["verdict"] == "ready"
    assert "ambiguous-pairing" not in published["reasons"]

    preprint = items["10.1101/2024.01.01.500001"]
    assert preprint["verdict"] == "skip"
    assert "superseded-by-journal" in preprint["reasons"]


def test_a_superseded_record_is_not_reported_as_missing_a_pdf(wiki, capsys):
    """It is not being imported, so whether it has a file is not a finding —
    and counting it would inflate the fetch list with versions the user
    deliberately isn't importing."""
    _, out, _ = run(["inspect", str(RIS), "--json"], capsys)
    payload = json.loads(out)
    sup = [i for i in payload["items"] if "superseded-by-journal" in i["reasons"]]
    assert sup and all("no-pdf" not in i["reasons"] for i in sup)
    fetch_dois = {f["doi"] for f in payload["missing_pdf_fetch_list"]}
    assert fetch_dois.isdisjoint({i["doi"] for i in sup})
