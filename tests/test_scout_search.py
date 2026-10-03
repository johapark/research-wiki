"""`scout search` / `scout search fetch`: leads, not evidence.

Pins the promises: sources fail independently (and the CLI still exits 2),
duplicate works merge across sources, what the wiki holds or the user
declined never resurfaces (under any of a lead's identifiers), abstracts and
trial summaries travel with the lead for triage, and fetch downloads only what
structured metadata says is open access — never over a file, never for a paper
already ingested.

Hermetic: providers, embedder and downloads are faked.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from researchwiki import __main__ as cli
from researchwiki.providers._http import (
    DownloadRefused,
    ProviderRequestRejected,
    StructuredProviderUnavailable,
)
from researchwiki.scouting import recent as R
from researchwiki.scouting import search as S
from researchwiki.scouting import search_cli as C
from researchwiki.scouting import search_fetch as F
from researchwiki.tasks import scout

TODAY = dt.date(2026, 10, 3)


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("wiki/cgt", "wiki/references", "wiki/synthesis", "inbox"):
        (tmp_path / name).mkdir(parents=True)
    # Unscored by default: ranking is `scout recent`'s, tested there.
    monkeypatch.setattr(R, "score_candidates", lambda cands, trig: True)
    return tmp_path


def _page(root, rel, **fm):
    lines = ["---"] + [f"{k}: {json.dumps(v)}" for k, v in fm.items()] + ["---", "", "## Summary\n\nx\n"]
    (root / "wiki" / f"{rel}.md").write_text("\n".join(lines), encoding="utf-8")


def _pm(pmid, doi=None, title="t", abstract="pubmed abstract", **kw):
    return {"pmid": pmid, "doi": doi, "pmcid": kw.get("pmcid"), "title": title,
            "authors": kw.get("authors", ["Baker A"]), "journal": "J",
            "pub_date": kw.get("date", "2026-09-01"), "year": kw.get("year", 2026),
            "pubtypes": [], "retracted": kw.get("retracted", False), "abstract": abstract}


def _ax(aid, title="t", journal_doi=None, abstract="arxiv abstract", authors=("Carol Dunn",)):
    return {"arxiv_id": aid, "version": "1", "doi": f"10.48550/arxiv.{aid}", "title": title,
            "authors": list(authors), "published": "2026-09-02", "updated": "2026-09-02",
            "year": 2026, "primary_category": "q-bio.GN", "journal_doi": journal_doi,
            "journal_ref": "", "abstract": abstract}


def _sources(monkeypatch, *, pubmed=(), arxiv=(), preprints=(), trials=(), fail=None):
    def make(name, rows):
        def call(query, **kw):
            if fail and name in fail:
                raise fail[name]
            return list(rows)
        return call
    monkeypatch.setattr(S.pubmed, "search", make("pubmed", pubmed))
    monkeypatch.setattr(S.arxiv, "search", make("arxiv", arxiv))
    monkeypatch.setattr(S.europepmc, "search_preprints", make("preprints", preprints))
    monkeypatch.setattr(S.clinicaltrials, "search", make("clinicaltrials", trials))


# ---------- merge and filter ----------

def test_journal_paper_and_its_arxiv_preprint_merge(wiki, monkeypatch):
    _sources(monkeypatch, pubmed=[_pm("1", doi="10.1038/x", title="Same Work")],
             arxiv=[_ax("2401.00001", title="Same work, preprint", journal_doi="10.1038/x")])
    snap = S.run("q", today=TODAY)
    [lead] = snap["leads"]
    assert lead["key"] == "10.1038/x"
    assert lead["sources"] == ["pubmed", "arxiv"]
    assert lead["fetch_key"] == "10.48550/arxiv.2401.00001"


def test_title_match_merges_records_without_shared_ids(wiki, monkeypatch):
    """PubMed `Dunn C` and arXiv `Carol Dunn` are one first author."""
    _sources(monkeypatch, pubmed=[_pm("1", title="Prime Editing: A Review.", authors=["Dunn C"])],
             arxiv=[_ax("2401.00002", title="prime editing a review")])
    snap = S.run("q", today=TODAY)
    assert len(snap["leads"]) == 1


def test_a_shared_title_alone_does_not_merge_distinct_papers(wiki, monkeypatch):
    _sources(monkeypatch, pubmed=[
        _pm("1", doi="10.1234/ed-one", title="Editorial", authors=["Smith J"]),
        _pm("2", doi="10.5678/ed-two", title="Editorial", authors=["Lee K"], year=2025),
        _pm("3", doi="10.9012/ed-three", title="Editorial", authors=["Smith J"]),
    ])
    snap = S.run("q", today=TODAY)
    assert sorted(r["key"] for r in snap["leads"]) == [
        "10.1234/ed-one", "10.5678/ed-two", "10.9012/ed-three"]


def test_shared_initials_are_not_a_shared_author(wiki, monkeypatch):
    """PubMed's `AV` initials block once counted as a matching name token."""
    _sources(monkeypatch, pubmed=[
        _pm("1", title="Editorial", authors=["Smith AV"]),
        _pm("2", title="Editorial", authors=["Jones AV"]),
    ])
    assert len(S.run("q", today=TODAY)["leads"]) == 2


def test_different_pmids_are_different_works_even_with_one_surname(wiki, monkeypatch):
    _sources(monkeypatch, pubmed=[
        _pm("1", title="Editorial", authors=["Smith AV"]),
        _pm("2", title="Editorial", authors=["Smith J"]),
    ])
    assert len(S.run("q", today=TODAY)["leads"]) == 2


def test_wiki_page_sharing_only_a_given_name_does_not_hide_a_lead(wiki, monkeypatch):
    _page(wiki, "cgt/smith-2026-editorial", title="Editorial", type="paper",
          doi="10.1234/held", authors="Carol Smith", year=2026)
    _sources(monkeypatch, arxiv=[_ax("2401.00003", title="Editorial", authors=("Carol Dunn",))])
    snap = S.run("q", sources=["arxiv"], today=TODAY)
    assert [r["key"] for r in snap["leads"]] == ["10.48550/arxiv.2401.00003"]


@pytest.mark.parametrize(("name", "expected"), [
    ("Smith AV", "smith"), ("Andrew V. Smith", "smith"), ("Andrew V Smith", "smith"),
    ("Dunn C", "dunn"), ("Carol Dunn", "dunn"),
    ("van der Berg JM", "berg"), ("Jan M van der Berg", "berg"),
    ("García-López A", "garcialopez"), ("Szałata A", "szalata"),
    ("Guohui Chuai et al.", "chuai"), ("Smith J Jr", "smith"),
    ("DeepSeek-AI", "deepseekai"), ("et al.", ""), ("", ""),
])
def test_surname_reads_both_name_orders(name, expected):
    assert S.surname(name) == expected


def test_title_match_refuses_contradicting_years_between_published_records(wiki, monkeypatch):
    _sources(monkeypatch, pubmed=[
        _pm("1", title="Annual review", authors=["Smith J"], year=2018),
        _pm("2", title="Annual review", authors=["Smith J"], year=2026),
    ])
    assert len(S.run("q", today=TODAY)["leads"]) == 2


def test_wiki_title_hides_only_the_corroborated_work(wiki, monkeypatch):
    _page(wiki, "cgt/smith-2026-editorial", title="Editorial", type="paper",
          doi="10.1234/held", authors="John Smith, Ann Lee", year=2026)
    _page(wiki, "cgt/dunn-2026-prime", title="Prime editing a review", type="paper",
          doi="10.1038/journal", authors="Carol Dunn", year=2026)
    _sources(monkeypatch,
             pubmed=[_pm("2", doi="10.5678/other", title="Editorial", authors=["Lee K"])],
             arxiv=[_ax("2401.00002", title="Prime Editing: A Review")])
    snap = S.run("q", today=TODAY)
    # A different journal DOI by a different author is a different paper; the
    # arXiv preprint of the held journal article is the same one.
    assert [r["key"] for r in snap["leads"]] == ["10.5678/other"]
    assert snap["counts"]["in_wiki"] == 1


def test_journal_page_retaining_arxiv_id_is_filtered(wiki, monkeypatch):
    _page(wiki, "cgt/a-2026-x", title="Published title", type="paper",
          doi="10.1234/journal", arxiv_id="2401.00002")
    _sources(monkeypatch, arxiv=[_ax("2401.00002", title="Different preprint title")])
    snap = S.run("q", sources=["arxiv"], today=TODAY)
    assert snap["leads"] == []
    assert snap["counts"]["in_wiki"] == 1


def test_unquoted_arxiv_id_still_counts_as_held(wiki, monkeypatch):
    """`arxiv_id: 2003.02320` unquoted is the float 2003.0232; the trailing
    zero must come back or the preprint resurfaces and fetch re-downloads it."""
    (wiki / "wiki" / "cgt" / "a-2020-x.md").write_text(
        "---\ntitle: Published title\ntype: paper\ndoi: 10.1234/journal\n"
        "arxiv_id: 2003.02320\n---\n\n## Summary\n\nx\n", encoding="utf-8")
    _sources(monkeypatch, arxiv=[_ax("2003.02320", title="Different preprint title")])
    snap = S.run("q", sources=["arxiv"], today=TODAY)
    assert snap["leads"] == [] and snap["counts"]["in_wiki"] == 1


@pytest.mark.parametrize(("value", "expected"), [
    (2003.0232, "2003.02320"),     # 5-digit era, one trailing zero lost
    (2510.102, "2510.10200"),      # two lost
    (2509.06917, "2509.06917"),    # nothing lost
    (1412.698, "1412.6980"),       # 4-digit era
    (704.0001, "0704.0001"),       # leading zero of a 2007 id lost too
    (912.1, "0912.1000"),          # leading and trailing
    ("2003.02320", "2003.02320"),  # quoted: verbatim
    ("hep-th/9901001", "hep-th/9901001"),
    (None, ""),
])
def test_arxiv_id_text_restores_yaml_float_ids(value, expected):
    assert S.arxiv_id_text(value) == expected


def test_in_wiki_and_declined_leads_never_resurface(wiki, monkeypatch):
    _page(wiki, "cgt/a-2026-x", title="Held Paper", type="paper", doi="10.1234/held",
          authors="Ann Baker", year=2026)
    _page(wiki, "references/nct-x", title="Protocol", type="protocol",
          document_id="NCT01234567")
    R.add_decline("pmid:7", "off-topic", source="search")
    R.add_decline("arxiv:2401.00009v3", "off-topic", source="search")
    _sources(
        monkeypatch,
        pubmed=[_pm("5", doi="10.1234/held", title="five"), _pm("6", title="Held paper"),
                _pm("7", doi="10.1234/new", title="seven"),
                _pm("8", doi="10.1234/keep", title="keep")],
        arxiv=[_ax("2401.00009", title="declined preprint")],
        trials=[{"nct_id": "NCT01234567", "brief_title": "held trial", "official_title": "",
                 "conditions": [], "interventions": [], "start_date": "2026-01",
                 **{k: None for k in ("overall_status", "phases", "study_type", "lead_sponsor",
                                      "primary_completion_date", "completion_date",
                                      "enrollment", "has_results", "references", "documents")}}],
    )
    snap = S.run("q", sources=list(S.SOURCES), today=TODAY)
    assert [r["key"] for r in snap["leads"]] == ["10.1234/keep"]
    assert snap["trials"] == []
    assert snap["counts"]["in_wiki"] == 3 and snap["counts"]["declined"] == 2


def test_since_drops_dated_leads_but_keeps_undated_ones(wiki, monkeypatch):
    _sources(monkeypatch, pubmed=[_pm("1", doi="10.1234/old", title="old", date="2020-01-01"),
                                  _pm("2", doi="10.1234/undated", title="undated", date="")])
    snap = S.run("q", since=dt.date(2026, 1, 1), today=TODAY)
    assert [r["key"] for r in snap["leads"]] == ["10.1234/undated"]
    assert snap["counts"]["outside_window"] == 1


# ---------- abstracts: triage text that travels with the lead ----------

def test_abstract_is_kept_with_the_lead(wiki, monkeypatch):
    _sources(monkeypatch, pubmed=[_pm("1", doi="10.1234/a", abstract="ABSTRACT-TEXT")])
    snap = S.run("q", today=TODAY)
    assert snap["leads"][0]["abstract"] == "ABSTRACT-TEXT"
    assert snap["leads"][0]["has_abstract"] is True
    on_disk = json.loads(open(snap["snapshot_path"], encoding="utf-8").read())
    assert on_disk["leads"][0]["abstract"] == "ABSTRACT-TEXT"


def test_abstract_is_embedded_for_ranking(wiki, monkeypatch):
    seen = []
    monkeypatch.setattr(R, "score_candidates",
                        lambda cands, trig: seen.extend(c.score_text for c in cands) or True)
    _sources(monkeypatch, pubmed=[_pm("1", doi="10.1234/a", title="T", abstract="ABS")])
    S.run("q", today=TODAY)
    assert seen == ["T\n\nABS"]


def test_terminal_shows_abstracts_on_request_and_json_always(wiki, monkeypatch, capsys):
    _sources(monkeypatch, pubmed=[_pm("1", doi="10.1234/a", abstract="VISIBLE-ABS")])
    assert C.main(["q", "--source", "pubmed"]) == 0
    assert "VISIBLE-ABS" not in capsys.readouterr().out
    assert C.main(["q", "--source", "pubmed", "--abstracts"]) == 0
    assert "abstract (pubmed, verbatim): VISIBLE-ABS" in capsys.readouterr().out
    assert C.main(["q", "--source", "pubmed", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["leads"][0]["abstract"] == "VISIBLE-ABS"


def test_trial_brief_summary_is_the_trial_abstract(wiki, monkeypatch, capsys):
    trial = {k: None for k in S.clinicaltrials.TRIAL_FIELDS}
    trial.update(nct_id="NCT07748403", brief_title="Prime editing in WD", official_title="",
                 brief_summary="TRIAL-SUMMARY", conditions=["Wilson Disease"],
                 interventions=[{"name": "PM577", "type": "GENETIC"}], start_date="2026-09",
                 phases=[], references=[], documents=[])
    _sources(monkeypatch, trials=[trial])
    assert C.main(["q", "--source", "clinicaltrials", "--abstracts"]) == 0
    assert "summary (clinicaltrials, verbatim): TRIAL-SUMMARY" in capsys.readouterr().out
    snap = S.run("q", sources=["clinicaltrials"], today=TODAY)
    [row] = snap["trials"]
    assert row["abstract"] == "TRIAL-SUMMARY" and "brief_summary" not in row["trial"]


# ---------- independent sources and exit codes ----------

def test_one_unavailable_source_still_prints_the_others_then_exits_2(wiki, monkeypatch, capsys):
    _sources(monkeypatch, pubmed=[_pm("1", doi="10.1234/a", title="kept")],
             fail={"arxiv": StructuredProviderUnavailable("arxiv API unavailable (HTTP 503)")})
    assert cli.main(["scout", "search", "q", "--source", "pubmed", "--source", "arxiv"]) == 2
    captured = capsys.readouterr()
    assert "kept" in captured.out
    assert "arxiv: UNAVAILABLE" in captured.out
    assert "arxiv unavailable" in captured.err


def test_rejected_query_exits_1(wiki, monkeypatch, capsys):
    _sources(monkeypatch, fail={"clinicaltrials": ProviderRequestRejected("HTTP 400: bad")})
    assert C.main(["q", "--source", "clinicaltrials", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["sources"]["clinicaltrials"]["status"] == "rejected"


def test_biorxiv_and_medrxiv_each_get_the_full_source_limit(wiki, monkeypatch):
    calls = []

    def preprints(query, *, servers, limit, **kw):
        [server] = servers
        calls.append((server, limit))
        return [{"doi": f"10.1101/{server}-{i}", "server": server,
                 "title": f"{server} result {i}"} for i in range(limit)]

    monkeypatch.setattr(S.europepmc, "search_preprints", preprints)
    leads, status = S.gather("q", ["biorxiv", "medrxiv"], limit=3,
                             since=None, max_age_days=1)
    assert calls == [("biorxiv", 3), ("medrxiv", 3)]
    assert len(leads) == 6
    assert status["biorxiv"]["returned"] == status["medrxiv"]["returned"] == 3


def test_preprint_source_failure_keeps_the_other_source(wiki, monkeypatch):
    def preprints(query, *, servers, **kw):
        if servers == ["biorxiv"]:
            raise StructuredProviderUnavailable("Europe PMC bioRxiv query failed")
        return [{"doi": "10.1101/med-1", "server": "medrxiv", "title": "Med lead"}]

    monkeypatch.setattr(S.europepmc, "search_preprints", preprints)
    leads, status = S.gather("q", ["biorxiv", "medrxiv"], limit=3,
                             since=None, max_age_days=1)
    assert [lead.title for lead in leads] == ["Med lead"]
    assert status["biorxiv"]["status"] == "unavailable"
    assert status["medrxiv"] == {"status": "ok", "returned": 1, "error": None}


@pytest.mark.parametrize("argv", [[], ["q", "--limit", "0"], ["q", "--since", "nope"],
                                  ["q", "--days", "0"], ["--decline", "10.1234/x"]])
def test_bad_arguments_exit_1(wiki, argv):
    assert C.main(argv) == 1


def test_query_option_allows_searching_for_the_word_fetch(wiki, monkeypatch):
    seen = []
    monkeypatch.setattr(S.pubmed, "search", lambda q, **kw: seen.append(q) or [])
    assert C.main(["--query", "fetch", "--source", "pubmed"]) == 0
    assert seen == ["fetch"]


# ---------- declines ----------

def test_search_declines_share_the_ledger_but_not_s2_negative_seeds(wiki, capsys):
    assert C.main(["--decline", "pmcid:PMC123", "--reason", "r"]) == 0
    assert "declined pmcid:pmc123" in capsys.readouterr().out
    assert C.main(["--decline", "10.1234/searched", "--reason", "r"]) == 0
    R.add_decline("10.1234/from-recent", "r")
    declines = R.load_declines()
    assert {"pmcid:pmc123", "10.1234/searched", "10.1234/from-recent"} <= set(declines)
    assert R.negative_dois(declines) == ["10.1234/from-recent"]
    assert C.main(["--undecline", "pmcid:pmc123"]) == 0


@pytest.mark.parametrize(("raw", "key"), [
    ("arxiv:2401.01234v2", "10.48550/arxiv.2401.01234"),
    ("https://arxiv.org/abs/2401.01234", "10.48550/arxiv.2401.01234"),
    ("arXiv:hep-th/9901001", "10.48550/arxiv.hep-th/9901001"),
    ("PMID:123", "pmid:123"),
    ("pmcid:123", "pmcid:pmc123"),
    ("NCT:NCT01234567", "nct:nct01234567"),
])
def test_search_key_forms_normalize(raw, key):
    assert R.normalize_key(raw) == key


@pytest.mark.parametrize("raw", ["123", "pmid:", "nct:nct123", "arxiv:", "pmcid:abc"])
def test_bare_or_malformed_search_keys_are_refused(raw):
    with pytest.raises(R.DeclineKeyError, match="not a DOI"):
        R.normalize_key(raw)


# ---------- fetch ----------

@pytest.fixture
def downloads(monkeypatch):
    got = []

    def fake(url, dest, **kw):
        assert kw["allowed_hosts"] == F.ALLOWED_HOSTS
        dest.write_bytes(b"%PDF-1.7")
        got.append(url)
        return dest
    monkeypatch.setattr(F, "curl_download", fake)
    return got


def test_fetch_arxiv_names_the_file_and_prints_the_capital_x_doi(wiki, downloads, capsys):
    assert C.main(["fetch", "arxiv:2401.01234v2"]) == 0
    assert downloads == ["https://export.arxiv.org/pdf/2401.01234"]
    assert (wiki / "inbox" / "arxiv-2401.01234.pdf").exists()
    out = capsys.readouterr().out
    assert "researchwiki agent ingest inbox/arxiv-2401.01234.pdf --doi 10.48550/arXiv.2401.01234" in out


def test_fetch_biorxiv_uses_latest_version_and_records_licence(wiki, downloads, monkeypatch):
    monkeypatch.setattr(F.biorxiv, "lookup", lambda doi: {
        "server": "medrxiv", "version": "3", "license": "cc_no"})
    report = F.fetch(["10.1101/2025.01.01.25300001"])
    [entry] = report["fetched"]
    assert downloads == ["https://www.medrxiv.org/content/10.1101/2025.01.01.25300001v3.full.pdf"]
    assert entry["license"] == "cc_no" and entry["ingest_doi"] == "10.1101/2025.01.01.25300001"


def test_fetch_refuses_closed_access_trials_and_ingested_papers(wiki, downloads, monkeypatch, capsys):
    _page(wiki, "cgt/a-2026-x", title="T", type="paper", doi="10.48550/arxiv.2401.00001")
    monkeypatch.setattr(F.europepmc, "oa_status", lambda key: {
        "is_open_access": False, "license": None, "pmid": "1", "pmcid": None,
        "doi": "10.1234/closed", "pdf_urls": []})
    assert C.main(["fetch", "10.1234/closed", "nct:NCT01234567", "arxiv:2401.00001", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    reasons = {e["key"]: e["reason"] for e in report["skipped"]}
    assert reasons["10.1234/closed"] == "not open access per Europe PMC"
    assert "trial" in reasons["nct:nct01234567"]
    assert reasons["10.48550/arxiv.2401.00001"] == "already in the wiki"
    assert downloads == []


def test_fetch_skips_arxiv_id_retained_by_journal_page(wiki, downloads):
    _page(wiki, "cgt/a-2026-x", title="Published title", type="paper",
          doi="10.1234/journal", arxiv_id="2401.00002")
    report = F.fetch(["arxiv:2401.00002"])
    assert report["skipped"] == [{"key": "10.48550/arxiv.2401.00002",
                                  "reason": "already in the wiki"}]
    assert report["fetched"] == [] and downloads == []


def test_fetch_skips_a_held_preprint_without_asking_any_provider(wiki, downloads, monkeypatch):
    """An outage must not stop the run over a paper it was never going to fetch."""
    _page(wiki, "cgt/w-2025-x", title="T", type="paper", doi="10.1101/2025.11.03.686307")

    def offline(*a, **k):
        raise StructuredProviderUnavailable("biorxiv API unavailable (offline)")
    monkeypatch.setattr(F.biorxiv, "lookup", offline)
    monkeypatch.setattr(F.europepmc, "oa_status", offline)
    report = F.fetch(["10.1101/2025.11.03.686307", "arxiv:2401.00001"])
    assert report["skipped"] == [{"key": "10.1101/2025.11.03.686307",
                                  "reason": "already in the wiki"}]
    assert report["stopped_on"] is None
    assert [e["key"] for e in report["fetched"]] == ["10.48550/arxiv.2401.00001"]


def test_fetch_never_overwrites_an_inbox_file(wiki, downloads):
    (wiki / "inbox" / "arxiv-2401.01234.pdf").write_bytes(b"%PDF-mine")
    report = F.fetch(["arxiv:2401.01234"])
    assert report["already_present"] and not report["fetched"] and downloads == []
    assert (wiki / "inbox" / "arxiv-2401.01234.pdf").read_bytes() == b"%PDF-mine"


def test_fetch_lists_a_refused_download_for_manual_retrieval(wiki, monkeypatch, capsys):
    monkeypatch.setattr(F.europepmc, "oa_status", lambda key: {
        "is_open_access": True, "license": "cc by", "pmid": "1", "pmcid": "PMC9",
        "doi": "10.1234/oa", "pdf_urls": ["https://europepmc.org/articles/PMC9?pdf=render"]})

    def refuse(url, dest, **kw):
        raise DownloadRefused("http-403", url)
    monkeypatch.setattr(F, "curl_download", refuse)
    assert C.main(["fetch", "pmcid:PMC9"]) == 1
    assert "download manually `pmcid:pmc9` (http-403): https://europepmc.org/articles/PMC9?pdf=render" \
        in capsys.readouterr().out
    assert list((wiki / "inbox").iterdir()) == []


def test_fetch_stops_at_the_first_outage_and_reports_what_landed(wiki, monkeypatch, capsys):
    calls = []

    def flaky(url, dest, **kw):
        calls.append(url)
        if len(calls) == 2:
            raise StructuredProviderUnavailable("network down")
        dest.write_bytes(b"%PDF-1.7")
        return dest
    monkeypatch.setattr(F, "curl_download", flaky)
    code = cli.main(["scout", "search", "fetch", "arxiv:2401.00001", "arxiv:2401.00002",
                     "arxiv:2401.00003"])
    assert code == 2
    assert len(calls) == 2
    out = capsys.readouterr().out
    assert "fetched `10.48550/arxiv.2401.00001`" in out
    assert "STOPPED at `arxiv:2401.00002`" in out


def test_fetch_dry_run_downloads_nothing(wiki, downloads):
    report = F.fetch(["arxiv:2401.01234"], dry_run=True)
    assert report["fetched"][0]["path"] == "inbox/arxiv-2401.01234.pdf"
    assert downloads == [] and list((wiki / "inbox").iterdir()) == []


def test_fetch_ingest_hands_one_batch_per_file_doi_overrides(wiki, downloads, monkeypatch):
    from researchwiki.tasks import _ingest_batch
    seen = {}
    monkeypatch.setattr(_ingest_batch, "new_batch",
                        lambda pdfs, sub, extra, workers, **kw: seen.update(pdfs=pdfs, sub=sub, **kw) or 0)
    monkeypatch.setattr(_ingest_batch, "resolve_batch_workers", lambda requested: 4)
    assert C.main(["fetch", "arxiv:2401.00001", "arxiv:2401.00002", "--ingest"]) == 0
    assert seen["sub"] == ["agent", "ingest"] and len(seen["pdfs"]) == 2
    assert sorted(v[1] for v in seen["per_input_args"].values()) == [
        "10.48550/arXiv.2401.00001", "10.48550/arXiv.2401.00002"]


# ---------- dispatch ----------

def test_scout_dispatches_search(monkeypatch):
    seen = []
    monkeypatch.setattr(scout.search_cli, "main", lambda argv: seen.append(argv) or 0)
    assert scout.main(["search", "prime", "editing"]) == 0
    assert seen == [["prime", "editing"]]
