"""`scout recent report`: merging every run's snapshot into one current view.

Pins the merge rules a dashboard will depend on: a re-run supersedes its own
seed set rather than merging with it, papers ingested or declined since a
snapshot was written drop out, a paper's matches accumulate across runs while
`first_seen` keeps the earliest sighting, and no abstract reaches the aggregate.

Hermetic: snapshots are written by hand into a temp `.s2-cache/recent/`; the
wiki is a few stub pages. No network, no model, no embedding index.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from researchwiki.scouting import recent as R
from researchwiki.scouting import recent_report as RR

TODAY = dt.date(2026, 9, 27)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("wiki/single-cell", "wiki/synthesis", "wiki/ideas", "wiki/other"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / ".s2-cache" / "recent").mkdir(parents=True)
    return tmp_path


def _page(root, rel, fm, body="## Summary\n\nx\n"):
    lines = ["---"] + [f"{k}: {json.dumps(v)}" for k, v in fm.items()] + ["---", "", body]
    (root / "wiki" / f"{rel}.md").write_text("\n".join(lines), encoding="utf-8")


def _paper_page(root, cat, stem, doi=None, title=None):
    fm = {"title": title or stem, "type": "paper", "year": 2026}
    if doi:
        fm["doi"] = doi
    _page(root, f"{cat}/{stem}", fm)


def _row(key, *, title=None, doi=None, paper_id=None, date="2026-09-01", fit=0.8,
         first_seen="2026-09-20", triggers=(), year=2026):
    if doi is None and not key.startswith("s2:"):
        doi = key            # real snapshots key a DOI paper by its DOI
    return {"key": key, "doi": doi, "paper_id": paper_id, "title": title or key,
            "publication_date": date, "year": year, "venue": "bioRxiv",
            "citation_count": 0, "has_abstract": True, "fit": fit,
            "nearest": [{"key": "single-cell/a-2026-x", "score": fit}],
            "triggers": list(triggers), "first_seen": first_seen}


def _trigger(page, z=3.0, text="A head-to-head benchmark of the two approaches."):
    return {"page": page, "page_type": "synthesis", "z": z, "score": 0.8,
            "neighbor_overlap": 3, "page_relevance": 0.9, "text": text}


def _snapshot(root, name, label, *, matches=(), candidates=(), generated_at="2026-09-27T09:00:00",
              version=R.SCHEMA_VERSION):
    snap = {
        "schema_version": version, "generated_at": generated_at,
        "provider": "semantic-scholar", "since": "2026-07-29",
        "seeds": {"label": label, "used": 3, "available": 3, "negatives": 0},
        "counts": {"returned": 10}, "scored": True,
        "trigger_gates": {"z_min": 2.25, "page_relevance_min": 0.85},
        "triggers_indexed": 5,
        "page_matches": list(matches), "candidates": list(candidates),
    }
    (root / ".s2-cache" / "recent" / name).write_text(json.dumps(snap), encoding="utf-8")
    return snap


# ---------- loading ----------

def test_no_snapshots_is_an_empty_report_not_a_crash(root):
    report = RR.aggregate(today=TODAY)
    assert report["counts"]["snapshots"] == 0
    assert report["papers"] == [] and report["by_page"] == []


def test_the_seen_ledger_and_unknown_versions_are_skipped(root):
    (root / ".s2-cache" / "recent" / RR.SEEN_FILENAME).write_text('{"10.1/x": "2026-09-01"}')
    _snapshot(root, "2026-09-27__aaa.json", "single-cell", candidates=[_row("10.1/keep")])
    _snapshot(root, "2026-09-27__bbb.json", "genomics", candidates=[_row("10.1/old")],
              version=R.SCHEMA_VERSION + 1)
    (root / ".s2-cache" / "recent" / "2026-09-27__ccc.json").write_text("{truncated")

    report = RR.aggregate(today=TODAY)
    assert report["counts"]["snapshots"] == 1
    assert report["counts"]["snapshots_skipped_version"] == 2
    assert [p["key"] for p in report["papers"]] == ["10.1/keep"]


def test_cli_exits_1_when_there_is_nothing_to_report(root, capsys):
    assert RR.main([]) == 1


# ---------- merge rules ----------

def test_a_rerun_supersedes_its_own_seed_set(root):
    """Newest snapshot per seed set wins. Merging them instead would resurrect a
    paper the re-run dropped, from its own predecessor."""
    _snapshot(root, "2026-09-20__aaa.json", "single-cell", generated_at="2026-09-20T09:00:00",
              candidates=[_row("10.1/gone"), _row("10.1/both")])
    _snapshot(root, "2026-09-27__aaa.json", "single-cell", generated_at="2026-09-27T09:00:00",
              candidates=[_row("10.1/both"), _row("10.1/new")])

    report = RR.aggregate(today=TODAY)
    assert report["counts"]["runs_used"] == 1 and report["counts"]["runs_superseded"] == 1
    assert sorted(p["key"] for p in report["papers"]) == ["10.1/both", "10.1/new"]


def test_different_seed_sets_merge_and_a_paper_keeps_every_page_it_matched(root):
    _page(root, "synthesis/s1", {"title": "S1", "type": "synthesis"})
    _page(root, "ideas/i1", {"title": "I1", "type": "idea"})
    _snapshot(root, "2026-09-27__aaa.json", "single-cell",
              matches=[_row("10.1/x", fit=0.70, triggers=[_trigger("synthesis/s1", z=2.5)])])
    _snapshot(root, "2026-09-27__bbb.json", "synthesis/s1",
              matches=[_row("10.1/x", fit=0.90, triggers=[_trigger("ideas/i1", z=3.4),
                                                          _trigger("synthesis/s1", z=2.9)])])

    report = RR.aggregate(today=TODAY)
    assert report["counts"]["runs_used"] == 2
    paper, = report["papers"]
    assert paper["runs"] == ["single-cell", "synthesis/s1"] or paper["runs"] == ["synthesis/s1", "single-cell"]
    assert [(m["page"], m["z"]) for m in paper["matches"]] == [("ideas/i1", 3.4), ("synthesis/s1", 2.9)]
    assert paper["fit"] == 0.90        # strongest evidence any run found
    assert paper["best_z"] == 3.4
    assert {b["page"] for b in report["by_page"]} == {"synthesis/s1", "ideas/i1"}


def test_first_seen_keeps_the_earliest_sighting(root):
    _snapshot(root, "2026-09-27__aaa.json", "a", candidates=[_row("10.1/x", first_seen="2026-09-25")])
    _snapshot(root, "2026-09-27__bbb.json", "b", candidates=[_row("10.1/x", first_seen="2026-09-10")])
    assert RR.aggregate(today=TODAY)["papers"][0]["first_seen"] == "2026-09-10"


def test_the_doi_check_falls_back_to_the_key(root):
    """A snapshot row whose `doi` field is absent but whose key is a DOI (an
    older build, a hand-edited cache) must still match the wiki."""
    _paper_page(root, "single-cell", "smith-2026-ingested", doi="10.1/ingested")
    _snapshot(root, "2026-09-27__aaa.json", "single-cell", candidates=[
        {**_row("10.1/ingested"), "doi": None}, _row("10.1/keep")])
    report = RR.aggregate(today=TODAY)
    assert [p["key"] for p in report["papers"]] == ["10.1/keep"]
    assert report["counts"]["dropped"]["ingested"] == 1


def test_a_paper_ingested_since_the_run_drops_out(root):
    """The snapshot is a record of what S2 returned then; the wiki has moved on."""
    _paper_page(root, "single-cell", "smith-2026-ingested", doi="10.1/ingested")
    _paper_page(root, "single-cell", "jones-2026-by-title", title="A Draft Human Pangenome Reference")
    _snapshot(root, "2026-09-27__aaa.json", "single-cell", candidates=[
        _row("10.1/ingested"),
        _row("10.9/preprint", title="A draft human pangenome reference."),
        _row("10.1/keep"),
    ])

    report = RR.aggregate(today=TODAY)
    assert [p["key"] for p in report["papers"]] == ["10.1/keep"]
    assert report["counts"]["dropped"]["ingested"] == 2


def test_a_decline_since_the_run_drops_out_including_after_s2_adds_a_doi(root):
    R.add_decline("10.1234/declined", "off-topic")
    R.add_decline("s2:abc", "no doi")
    _snapshot(root, "2026-09-27__aaa.json", "single-cell", candidates=[
        _row("10.1234/declined", doi="10.1234/declined"),
        _row("10.1234/late-doi", doi="10.1234/late-doi", paper_id="ABC"),   # declined as s2:abc
        _row("10.1/keep"),
    ])

    report = RR.aggregate(today=TODAY)
    assert [p["key"] for p in report["papers"]] == ["10.1/keep"]
    assert report["counts"]["dropped"]["declined"] == 2


def test_a_match_to_a_deleted_page_is_dropped(root):
    _snapshot(root, "2026-09-27__aaa.json", "single-cell",
              matches=[_row("10.1/x", triggers=[_trigger("synthesis/removed")])])
    report = RR.aggregate(today=TODAY)
    assert report["papers"][0]["matches"] == []
    assert report["counts"]["dropped"]["page_gone"] == 1


def test_the_window_is_re_applied_at_report_time(root):
    _snapshot(root, "2026-09-27__aaa.json", "single-cell", candidates=[
        _row("10.1/fresh", date="2026-09-20"),
        _row("10.1/stale", date="2026-08-01"),
        _row("10.1/undated", date=None, year=2026),
    ])
    wide = RR.aggregate(today=TODAY)
    assert len(wide["papers"]) == 3 and wide["window"]["since"] is None

    narrow = RR.aggregate(days=14, today=TODAY)
    assert sorted(p["key"] for p in narrow["papers"]) == ["10.1/fresh", "10.1/undated"]
    assert narrow["counts"]["dropped"]["outside_window"] == 1
    assert narrow["window"] == {"since": "2026-09-13", "days": 14}


def test_matched_papers_sort_before_unmatched_and_by_strength(root):
    _page(root, "synthesis/s1", {"title": "S1", "type": "synthesis"})
    _snapshot(root, "2026-09-27__aaa.json", "single-cell",
              matches=[_row("10.1/weak", triggers=[_trigger("synthesis/s1", z=2.4)]),
                       _row("10.1/strong", triggers=[_trigger("synthesis/s1", z=3.9)])],
              candidates=[_row("10.1/plain", fit=0.99)])
    assert [p["key"] for p in RR.aggregate(today=TODAY)["papers"]] == [
        "10.1/strong", "10.1/weak", "10.1/plain"]


# ---------- contract ----------

def test_the_aggregate_carries_no_abstract(root):
    _snapshot(root, "2026-09-27__aaa.json", "single-cell",
              candidates=[{**_row("10.1/x"), "abstract": "VERBATIM ABSTRACT TEXT"}])
    blob = json.dumps(RR.aggregate(today=TODAY))
    assert "VERBATIM ABSTRACT TEXT" not in blob
    assert '"abstract"' not in blob
    assert '"has_abstract"' in blob


def test_the_json_shape_is_the_documented_contract(root):
    _page(root, "synthesis/s1", {"title": "S1", "type": "synthesis"})
    _snapshot(root, "2026-09-27__aaa.json", "single-cell",
              matches=[_row("10.1/x", doi="10.1/x", triggers=[_trigger("synthesis/s1")])])
    report = RR.aggregate(today=TODAY)
    assert report["schema_version"] == RR.REPORT_SCHEMA_VERSION
    assert set(report) == {"schema_version", "generated_at", "window", "counts", "runs",
                           "trigger_gates", "papers", "by_page"}
    assert set(report["counts"]) == {"snapshots", "runs_used", "runs_superseded",
                                     "snapshots_skipped_version", "papers",
                                     "papers_with_matches", "pages_with_papers", "dropped"}
    paper = report["papers"][0]
    assert set(paper) == {"key", "doi", "paper_id", "title", "venue", "publication_date",
                          "year", "citation_count", "has_abstract", "fit", "first_seen",
                          "nearest", "matches", "runs", "best_z", "decline_command"}
    assert paper["decline_command"] == 'researchwiki scout recent --decline 10.1/x --reason "…"'
    assert set(report["by_page"][0]) == {"page", "page_type", "papers"}


def test_a_doi_less_paper_gets_an_s2_url_decline_command(root):
    _snapshot(root, "2026-09-27__aaa.json", "single-cell",
              candidates=[_row("s2:abc", paper_id="abc")])
    cmd = RR.aggregate(today=TODAY)["papers"][0]["decline_command"]
    assert cmd.endswith('--reason "…"') and "semanticscholar.org/paper/abc" in cmd
    # and that URL is a key the CLI accepts
    assert R.normalize_key("https://www.semanticscholar.org/paper/abc") == "s2:abc"


def test_no_model_is_ever_called(root, monkeypatch):
    from researchwiki.agents import llm
    monkeypatch.setattr(llm, "call", lambda **kw: pytest.fail("report called a model"))
    _snapshot(root, "2026-09-27__aaa.json", "single-cell", candidates=[_row("10.1/x")])
    RR.aggregate(today=TODAY)


# ---------- rendering / CLI ----------

def test_render_shows_matches_first_and_groups_by_page(root):
    _page(root, "synthesis/s1", {"title": "S1", "type": "synthesis"})
    _snapshot(root, "2026-09-27__aaa.json", "single-cell",
              matches=[_row("10.1/m", title="Matched", triggers=[_trigger("synthesis/s1")])],
              candidates=[_row("10.1/p", title="Plain")])
    out = RR.render(RR.aggregate(today=TODAY), 10)
    assert out.index("Matched") < out.index("Plain")
    assert "on-topic for [[synthesis/s1]] (z 3.0)" in out
    assert "## By page" in out and "- [[synthesis/s1]] — 1 paper(s)" in out
    assert "Leads only" in out


def test_render_limit_caps_both_sections(root):
    _page(root, "synthesis/s1", {"title": "S1", "type": "synthesis"})
    _snapshot(root, "2026-09-27__aaa.json", "single-cell",
              matches=[_row(f"10.1/m{i}", title=f"Match {i}",
                            triggers=[_trigger("synthesis/s1", z=3.0 - i * 0.1)]) for i in range(3)],
              candidates=[_row(f"10.1/p{i}", title=f"Plain {i}") for i in range(3)])
    out = RR.render(RR.aggregate(today=TODAY), 2)
    assert "Match 0" in out and "Match 1" in out and "Match 2" not in out
    assert "Plain 0" not in out
    assert "Near a page's open questions (2 of 3)" in out


def test_cli_writes_json_or_text_to_a_file(root, capsys):
    _snapshot(root, "2026-09-27__aaa.json", "single-cell", candidates=[_row("10.1/x")])
    assert RR.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == RR.REPORT_SCHEMA_VERSION

    out = root / "output" / "scout-recent.md"
    assert RR.main(["--out", str(out)]) == 0
    assert "# Papers worth a look" in out.read_text()
    assert out.read_text().endswith("\n")


def test_cli_rejects_a_bad_window(root, capsys):
    _snapshot(root, "2026-09-27__aaa.json", "single-cell", candidates=[_row("10.1/x")])
    assert RR.main(["--since", "2026-13-40"]) == 1
    assert "invalid --since" in capsys.readouterr().err
    assert RR.main(["--days", "0"]) == 1


def test_scout_recent_dispatches_report(monkeypatch):
    seen = []
    from researchwiki.scouting import recent_report
    monkeypatch.setattr(recent_report, "main", lambda argv: seen.append(argv) or 0)
    assert R.main(["report", "--json"]) == 0
    assert seen == [["--json"]]
