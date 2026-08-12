"""`researchwiki export` end to end.

Pages are built inline rather than from a fixture directory: what these tests
need is one page per *policy branch*, and spelling them out here keeps each test
readable next to the rule it pins. The escaping and name rules are covered from
strings in `tests/test_refexport.py`.
"""

import json

import pytest

from researchwiki.refimport.parse import parse_bibtex, parse_csl_json, parse_ris
from researchwiki.tasks import export as export_task


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    """A wiki root. Required: `__main__` refuses any command but `init` when
    `./wiki` is absent, so without the chdir every assertion would measure that
    guard instead of the code under test."""
    for d in ("cgt", "ai", "references", "synthesis", "ideas", "concepts"):
        (tmp_path / "wiki" / d).mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def add_page(wiki, category, stem, **fields):
    fm = {"title": "A paper about things", "type": "paper",
          "authors": "Ada Fixture, Brian Second", "year": 2024,
          "doi": "10.1234/a", "venue": "Nature"}
    fm.update(fields)
    lines = ["---"]
    for k, v in fm.items():
        if v is None:
            continue
        lines.append(f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}")
    lines += ["---", "", "## Summary", "", "Body."]
    path = wiki / "wiki" / category / f"{stem}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(argv, capsys):
    code = export_task.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


# ---------- what is exportable ----------

def test_only_bibliographic_page_types_are_exported(wiki, capsys):
    add_page(wiki, "cgt", "bae-2014-a-paper")
    add_page(wiki, "references", "fda-2026-guidance", type="guidance",
             doi=None, venue=None, issuer="FDA", issuance_date="April 2026")
    add_page(wiki, "synthesis", "off-target-strategies", type="synthesis",
             doi=None, venue=None, authors=None)
    add_page(wiki, "ideas", "an-idea", type="idea", doi=None, venue=None,
             authors=None)
    add_page(wiki, "concepts", "a-concept", type="concept", doi=None,
             venue=None, authors=None)

    code, out, err = run([], capsys)
    assert code == 0
    keys = {i.key for i in parse_bibtex(out)}
    assert keys == {"bae-2014-a-paper", "fda-2026-guidance"}


def test_a_synthesis_page_is_never_emitted_even_as_the_only_page(wiki, capsys):
    """An entry for a synthesis page would assert, once pasted into a manuscript,
    a publication that does not exist. That is a citation-integrity harm, so
    there is no flag for it."""
    add_page(wiki, "synthesis", "off-target-strategies", type="synthesis",
             doi=None, venue=None, authors=None)
    code, out, err = run([], capsys)
    assert code == 0 and out == ""
    assert "0 records" in err


# ---------- the gap policy ----------

def test_a_paper_with_no_venue_is_downgraded_not_skipped(wiki, capsys):
    """`@article` with no `journal` makes bibtex merely warn, surfacing weeks
    later in a LaTeX log. `@misc` surfaces in the bibliography and the report."""
    add_page(wiki, "cgt", "bae-2014-a-paper", venue=None)
    code, out, err = run(["--json"], capsys)
    report = json.loads(out)
    assert report["venue_missing"] == ["bae-2014-a-paper"]
    assert report["by_entry_type"] == {"misc": 1}
    assert report["records"] == 1, "downgraded, not dropped"


def test_a_furniture_venue_is_suppressed(wiki, capsys):
    """The one place this command could print a falsehood: a masthead artifact
    recorded as the journal."""
    add_page(wiki, "cgt", "wang-2024-a-paper",
             venue="Journal of LaTeX Class Files")
    code, out, err = run(["--json"], capsys)
    report = json.loads(out)
    assert report["venue_furniture"] == [
        {"stem": "wang-2024-a-paper", "venue": "Journal of LaTeX Class Files"}]

    _, bib, _ = run([], capsys)
    assert "LaTeX Class Files" not in bib
    assert parse_bibtex(bib)[0].venue is None


def test_a_doi_less_paper_is_emitted_with_its_reason(wiki, capsys):
    """No format requires a DOI, and `no_doi_reason` is human-authored prose
    explaining a citation gap — exactly what a reader of the `.bib` wants."""
    add_page(wiki, "cgt", "liao-2025-a-paper", doi=None,
             no_doi_reason="NeurIPS poster — no DOI assigned")
    code, out, err = run([], capsys)
    assert code == 0
    (item,) = parse_bibtex(out)
    assert item.doi is None
    assert "NeurIPS poster" in out


def test_a_prose_byline_yields_an_entry_with_no_author(wiki, capsys):
    add_page(wiki, "references", "anthropic-2026-a-note", type="whitepaper",
             doi=None, venue=None,
             authors=("Laura Luebbert (Anthropic Science). Based on research by "
                      "Ferdous Nasri, Sarah Gurev, Patrick Varilly"))
    code, out, err = run(["--json"], capsys)
    assert json.loads(out)["authors_unparseable"] == ["anthropic-2026-a-note"]
    _, bib, _ = run([], capsys)
    assert "author = " not in bib


def test_a_year_comes_from_issuance_date_when_absent(wiki, capsys):
    """Every `guidance` page in the corpus has an `issuance_date` and no `year:`."""
    add_page(wiki, "references", "fda-2026-guidance", type="guidance",
             year=None, doi=None, venue=None, issuer="FDA",
             issuance_date="April 2026")
    code, out, err = run([], capsys)
    assert parse_bibtex(out)[0].year == 2026


def test_the_report_accounts_for_every_selected_page(wiki, capsys):
    """`records + skipped` is what makes "report, don't guess" checkable."""
    add_page(wiki, "cgt", "bae-2014-a-paper")
    add_page(wiki, "cgt", "no-title-here", title=None)
    report = json.loads(run(["--json"], capsys)[1])
    assert report["records"] == 1
    assert report["skipped"] == [{"stem": "no-title-here", "reason": "no title"}]


# ---------- filters ----------

def test_category_filter(wiki, capsys):
    add_page(wiki, "cgt", "bae-2014-a-paper")
    add_page(wiki, "ai", "asai-2023-a-paper")
    keys = {i.key for i in parse_bibtex(run(["--category", "cgt"], capsys)[1])}
    assert keys == {"bae-2014-a-paper"}


def test_year_range_is_inclusive(wiki, capsys):
    for year in (2022, 2024, 2026):
        add_page(wiki, "cgt", f"bae-{year}-a-paper", year=year,
                 doi=f"10.1234/{year}")
    keys = {i.key for i in parse_bibtex(run(["--year", "2024-2026"], capsys)[1])}
    assert keys == {"bae-2024-a-paper", "bae-2026-a-paper"}


def test_a_reversed_year_range_is_a_user_error(wiki, capsys):
    """Same refusal as `db papers`, so the two cannot disagree about what a range
    selects."""
    code, out, err = run(["--year", "2026-2020"], capsys)
    assert code == 1
    assert "reversed" in err and "2020-2026" in err


def test_a_malformed_year_is_a_user_error(wiki, capsys):
    assert run(["--year", "recent"], capsys)[0] == 1


def test_stem_filter_selects_exactly_those_pages(wiki, capsys):
    add_page(wiki, "cgt", "bae-2014-a-paper")
    add_page(wiki, "cgt", "kim-2019-a-paper", doi="10.1234/b")
    keys = {i.key for i in parse_bibtex(run(["--stem", "kim-2019-a-paper"], capsys)[1])}
    assert keys == {"kim-2019-a-paper"}


def test_zero_matches_is_success(wiki, capsys):
    """A filter matching nothing is a result, not a failure — same contract as
    `db papers`."""
    add_page(wiki, "cgt", "bae-2014-a-paper")
    code, out, err = run(["--category", "nope"], capsys)
    assert code == 0 and out == ""
    assert "0 records" in err


# ---------- output plumbing ----------

def test_stdout_carries_only_the_bibliography(wiki, capsys):
    """So it can be piped. The summary is on stderr, which is what makes
    `> refs.bib` produce a clean file."""
    add_page(wiki, "cgt", "bae-2014-a-paper")
    code, out, err = run([], capsys)
    assert parse_bibtex(out)                       # stdout parses as BibTeX
    assert "records" in err                        # summary went to stderr


def test_json_claims_stdout_and_suppresses_the_bibliography(wiki, capsys):
    """Two payloads on stdout and neither could be piped."""
    add_page(wiki, "cgt", "bae-2014-a-paper")
    code, out, err = run(["--json"], capsys)
    json.loads(out)
    assert "@" not in out


def test_out_writes_the_file_and_reports_on_stdout_free_stderr(wiki, capsys):
    add_page(wiki, "cgt", "bae-2014-a-paper")
    code, out, err = run(["--out", "refs.bib"], capsys)
    assert code == 0
    assert (wiki / "refs.bib").read_text(encoding="utf-8").startswith("@article{")
    assert out == "", "the payload went to the file, not stdout"


def test_json_with_out_gives_both(wiki, capsys):
    add_page(wiki, "cgt", "bae-2014-a-paper")
    code, out, err = run(["--json", "--out", "refs.bib"], capsys)
    assert json.loads(out)["records"] == 1
    assert (wiki / "refs.bib").exists()


def test_a_missing_out_directory_is_created(wiki, capsys):
    """`write_text_atomic` mkdirs the parent, so `--out reports/refs.bib` works
    without a setup step. Pinned because the alternative — erroring — would be a
    reasonable-looking change that breaks a documented invocation."""
    add_page(wiki, "cgt", "bae-2014-a-paper")
    assert run(["--out", "reports/refs.bib"], capsys)[0] == 0
    assert (wiki / "reports" / "refs.bib").exists()


def test_an_unwritable_out_path_is_an_environment_failure(wiki, capsys):
    """Exit 2, not a traceback: the disk is the problem, not the argument. A file
    where a directory is needed is the portable way to be genuinely unwritable."""
    from researchwiki.errors import EnvironmentFailure
    add_page(wiki, "cgt", "bae-2014-a-paper")
    (wiki / "blocker").write_text("not a directory", encoding="utf-8")
    with pytest.raises(EnvironmentFailure):
        export_task.main(["--out", "blocker/refs.bib"])
    capsys.readouterr()


def test_output_is_byte_identical_across_runs(wiki, capsys):
    add_page(wiki, "cgt", "bae-2014-a-paper")
    add_page(wiki, "ai", "asai-2023-a-paper", doi="10.1234/b")
    first = run([], capsys)[1]
    second = run([], capsys)[1]
    assert first == second


# ---------- the released --json contract ----------

def test_json_keys_are_stable(wiki, capsys):
    """Exact set equality: removing a key is breaking (`CHANGELOG.md`), and
    adding one is a deliberate edit here rather than an unnoticed side effect."""
    add_page(wiki, "cgt", "bae-2014-a-paper")
    report = json.loads(run(["--json"], capsys)[1])
    assert set(report) == {
        "format", "records", "by_entry_type", "venue_missing",
        "venue_furniture", "doi_missing", "authors_unparseable", "skipped",
    }


# ---------- round-trip ----------

@pytest.mark.parametrize("fmt,parse", [
    ("bibtex", parse_bibtex), ("ris", parse_ris), ("csl-json", parse_csl_json),
])
def test_round_trips_through_our_own_importer(wiki, capsys, fmt, parse):
    """Also verified over the whole real corpus: 421 records, three formats, zero
    mismatches on title, authors, DOI, venue or year."""
    add_page(wiki, "cgt", "bae-2014-a-paper",
             title="CRISPR-Cas9 editing at 50% efficiency",
             authors="A. van der Graaf, Christopher Ré")
    code, out, err = run(["--format", fmt], capsys)
    (item,) = parse(out)
    assert item.key == "bae-2014-a-paper"
    assert item.title == "CRISPR-Cas9 editing at 50% efficiency"
    assert item.authors == ["A. van der Graaf", "Christopher Ré"]
    assert item.doi == "10.1234/a"
    assert item.venue == "Nature"
    assert item.year == 2024


def test_absent_fields_are_absent_not_empty(wiki, capsys):
    """The corpus carries no volume, issue, pages, publisher, ISSN or abstract on
    any page. An empty `pages = {}` is worse than no field."""
    add_page(wiki, "cgt", "bae-2014-a-paper")
    out = run([], capsys)[1]
    for field in ("volume", "number", "pages", "publisher", "issn", "abstract"):
        assert f"{field} = " not in out
