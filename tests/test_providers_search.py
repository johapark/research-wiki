"""`scout search` providers: fixed schemas and error shapes.

Each parser is fed an inline payload shaped like the real API's (probed
2026-10-03) and must emit exactly its declared fields, so a provider change
that adds or drops a key is a visible, reviewed edit. Hermetic — transports
are faked.
"""

from __future__ import annotations

import json

import pytest

from researchwiki.providers import (
    _http,
    arxiv,
    biorxiv,
    clinicaltrials,
    europepmc,
    openfda,
    pubmed,
)
from researchwiki.providers._http import ProviderRequestRejected, StructuredProviderUnavailable

EFETCH = """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2025//EN" "x.dtd">
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID>31634902</PMID>
   <Article>
    <Journal><Title>Nature</Title>
     <JournalIssue><PubDate><Year>2019</Year><Month>Dec</Month></PubDate></JournalIssue>
    </Journal>
    <ArticleTitle>Search-and-replace genome editing without <i>double-strand</i> breaks.</ArticleTitle>
    <Abstract>
     <AbstractText Label="BACKGROUND">Most genetic variants.</AbstractText>
     <AbstractText>Here we describe prime editing.</AbstractText>
    </Abstract>
    <AuthorList>
     <Author><LastName>Anzalone</LastName><Initials>AV</Initials></Author>
     <Author><CollectiveName>Some Consortium</CollectiveName></Author>
    </AuthorList>
    <PublicationTypeList>
     <PublicationType>Journal Article</PublicationType>
     <PublicationType>Retracted Publication</PublicationType>
    </PublicationTypeList>
    <ArticleDate DateType="Electronic"><Year>2019</Year><Month>10</Month><Day>21</Day></ArticleDate>
   </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList>
   <ArticleId IdType="pubmed">31634902</ArticleId>
   <ArticleId IdType="doi">10.1038/S41586-019-1711-4</ArticleId>
   <ArticleId IdType="pmc">pmc6907074</ArticleId>
  </ArticleIdList></PubmedData>
 </PubmedArticle>
</PubmedArticleSet>"""

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
 <entry>
  <id>http://arxiv.org/abs/2201.07338v2</id>
  <updated>2022-06-01T00:00:00Z</updated>
  <published>2022-01-18T00:00:00Z</published>
  <title>Controllable Protein
    Design with Language Models</title>
  <summary>  We review protein language models. </summary>
  <author><name>Noelia Ferruz</name></author>
  <author><name>Birte Hocker</name></author>
  <arxiv:doi>10.1038/S42256-022-00499-Z</arxiv:doi>
  <arxiv:journal_ref>Nat Mach Intell 4 (2022)</arxiv:journal_ref>
  <arxiv:comment>40 pages, prose we never read</arxiv:comment>
  <arxiv:primary_category term="q-bio.QM"/>
 </entry>
</feed>"""

ATOM_ERROR = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>http://arxiv.org/api/errors#incorrect_id_format_for_x</id>
  <title>Error</title>
  <summary>incorrect id format for x</summary>
 </entry>
</feed>"""


@pytest.fixture
def cache(tmp_path, monkeypatch):
    for mod in (pubmed, arxiv, europepmc, clinicaltrials, openfda):
        monkeypatch.setattr(mod, "web_cache_dir", lambda: tmp_path)
        monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    return tmp_path


# ---------- PubMed ----------

def test_efetch_parse_is_the_declared_schema():
    [rec] = pubmed.parse_efetch(EFETCH)
    assert set(rec) == set(pubmed.SEARCH_FIELDS)
    assert rec["pmid"] == "31634902"
    assert rec["doi"] == "10.1038/s41586-019-1711-4"
    assert rec["pmcid"] == "PMC6907074"
    assert rec["title"] == "Search-and-replace genome editing without double-strand breaks"
    assert rec["authors"] == ["Anzalone AV", "Some Consortium"]
    assert rec["first_author_surname"] == "Anzalone"
    assert rec["pub_date"] == "2019-10-21"  # electronic date wins over the issue date
    assert rec["year"] == 2019
    assert rec["retracted"] is True
    assert rec["abstract"] == "BACKGROUND: Most genetic variants.\nHere we describe prime editing."


def test_efetch_rejects_entity_declarations():
    evil = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>'
    with pytest.raises(StructuredProviderUnavailable, match="entities"):
        pubmed.parse_efetch(evil)


def test_pubmed_search_keeps_relevance_order_and_hides_secrets(cache, monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", "sekrit-key")
    monkeypatch.setenv("RW_CONTACT_EMAIL", "me@example.org")
    seen = []

    def body(url, *, secret_query=None, retries=3):
        seen.append((url, secret_query))
        if "/esearch.fcgi" in url:
            return json.dumps({"esearchresult": {"idlist": ["99", "31634902"]}})
        return EFETCH

    monkeypatch.setattr(pubmed, "_curl_body", body)
    recs = pubmed.search("prime editing", limit=2)
    assert [r["pmid"] for r in recs] == ["31634902"]  # 99 absent from efetch, dropped
    for url, secret in seen:
        assert "sekrit" not in url and "example.org" not in url
        assert secret == {"api_key": "sekrit-key", "email": "me@example.org"}
    assert not any("sekrit" in p.name for p in cache.rglob("*"))


# ---------- arXiv ----------

def test_atom_parse_is_the_declared_schema():
    [rec] = arxiv.parse_feed(ATOM)
    assert set(rec) == set(arxiv.SEARCH_FIELDS)
    assert rec["arxiv_id"] == "2201.07338" and rec["version"] == "2"
    assert rec["doi"] == "10.48550/arxiv.2201.07338"
    assert rec["journal_doi"] == "10.1038/s42256-022-00499-z"
    assert rec["title"] == "Controllable Protein Design with Language Models"
    assert rec["abstract"] == "We review protein language models."
    assert "prose we never read" not in json.dumps(rec)


def test_arxiv_error_entry_is_a_rejected_query_not_a_paper():
    with pytest.raises(ProviderRequestRejected, match="incorrect id format"):
        arxiv.parse_feed(ATOM_ERROR)


@pytest.mark.parametrize(("query", "expected"), [
    ("protein language model", "all:protein AND all:language AND all:model"),
    ('"prime editing" OR base', 'all:"prime editing" OR all:base'),
    ("ti:transformer AND cat:cs.LG", "ti:transformer AND cat:cs.LG"),
])
def test_arxiv_query_building(query, expected):
    assert arxiv.build_query(query) == expected


def test_arxiv_since_adds_a_submitted_date_window():
    import datetime as dt
    q = arxiv.build_query("x", dt.date(2026, 1, 2))
    assert q.startswith("(all:x) AND submittedDate:[202601020000 TO ")


# ---------- Europe PMC ----------

EPMC_PREPRINT = {"hitCount": 1, "resultList": {"result": [{
    "id": "PPR1", "source": "PPR", "doi": "10.1101/2025.11.03.686307",
    "title": "Generating  long deletions", "firstPublicationDate": "2025-11-04",
    "pubYear": "2025", "abstractText": "We built a screen.",
    "authorList": {"author": [{"fullName": "Weller J", "lastName": "Weller", "firstName": "Juliane"}]},
    "bookOrReportDetails": {"publisher": "bioRxiv"},
    "isOpenAccess": "N", "citedByCount": 3,
}]}}


def test_preprint_search_schema_and_query(cache, monkeypatch):
    urls = []
    monkeypatch.setattr(europepmc, "_curl_json", lambda url, retries=3: urls.append(url) or EPMC_PREPRINT)
    [rec] = europepmc.search_preprints("long deletions", servers=["biorxiv", "medrxiv"], limit=5)
    assert set(rec) == set(europepmc.SEARCH_FIELDS)
    assert rec["server"] == "biorxiv" and rec["title"] == "Generating long deletions"
    assert rec["first_author_surname"] == "Weller"
    assert "SRC%3APPR" in urls[0] and "medRxiv" in urls[0]


def test_preprint_surname_falls_back_to_the_vancouver_author_string():
    assert europepmc._first_surname({"authorString": "van der Berg JM, Lee K."}) == "van der Berg"
    assert europepmc._first_surname({"authorString": "Menendez A"}) == "Menendez"
    assert europepmc._first_surname({}) == ""


def test_europepmc_busy_body_is_an_outage(cache, monkeypatch):
    monkeypatch.setattr(europepmc, "_curl_json",
                        lambda url, retries=3: {"errCode": 503, "errMsg": "busy"})
    with pytest.raises(StructuredProviderUnavailable, match="busy"):
        europepmc.search_preprints("x", servers=["biorxiv"])


def test_oa_status_reads_only_open_access_pdf_links(cache, monkeypatch):
    payload = {"resultList": {"result": [{
        "pmid": "31634902", "pmcid": "PMC6907074", "doi": "10.1038/X", "isOpenAccess": "Y",
        "fullTextUrlList": {"fullTextUrl": [
            {"availabilityCode": "S", "documentStyle": "pdf", "url": "https://paywall/x.pdf"},
            {"availabilityCode": "OA", "documentStyle": "html", "url": "https://europepmc.org/a"},
            {"availabilityCode": "OA", "documentStyle": "pdf", "url": "https://europepmc.org/a?pdf=render"},
        ]},
    }]}}
    monkeypatch.setattr(europepmc, "_curl_json", lambda url, retries=3: payload)
    oa = europepmc.oa_status("pmcid:pmc6907074")
    assert set(oa) == set(europepmc.OA_FIELDS)
    assert oa["is_open_access"] is True and oa["doi"] == "10.1038/x"
    assert oa["pdf_urls"] == ["https://europepmc.org/a?pdf=render"]


# ---------- ClinicalTrials.gov ----------

STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT05066165", "briefTitle": "Trial",
                                 "officialTitle": "A Trial"},
        "statusModule": {"overallStatus": "RECRUITING", "startDateStruct": {"date": "2021-12"}},
        "designModule": {"phases": ["PHASE1"], "studyType": "INTERVENTIONAL",
                         "enrollmentInfo": {"count": 12}},
        "conditionsModule": {"conditions": ["ATTR"]},
        "armsInterventionsModule": {"interventions": [{"type": "GENETIC", "name": "NTLA-2001"}]},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Intellia"}},
        "referencesModule": {"references": [{"pmid": "34215024", "type": "RESULT"}]},
        "descriptionModule": {"briefSummary": "A phase 1 study.",
                              "detailedDescription": "long text we do not request"},
    },
    "documentSection": {"largeDocumentModule": {"largeDocs": [
        {"hasProtocol": True, "hasSap": False, "label": "Study Protocol", "filename": "Prot_001.pdf"},
    ]}},
    "hasResults": False,
}


def test_trial_parse_is_the_declared_schema_without_prose():
    rec = clinicaltrials.parse_study(STUDY)
    assert set(rec) == set(clinicaltrials.TRIAL_FIELDS)
    assert rec["documents"][0]["url"] == (
        "https://cdn.clinicaltrials.gov/large-docs/65/NCT05066165/Prot_001.pdf")
    assert rec["references"] == [{"pmid": "34215024", "type": "RESULT"}]
    assert rec["brief_summary"] == "A phase 1 study."
    assert "long text we do not request" not in json.dumps(rec)


def test_trial_fields_skip_the_long_registry_prose():
    """The brief summary is triage text; the rest is long and not requested."""
    assert "BriefSummary" in clinicaltrials.API_FIELDS
    assert not {"DetailedDescription", "EligibilityCriteria", "ReferenceCitation"} \
        & set(clinicaltrials.API_FIELDS)


def test_trial_query_rejection_propagates(cache, monkeypatch):
    def reject(url, retries=3):
        raise ProviderRequestRejected("clinicaltrials rejected the request (HTTP 400)")
    monkeypatch.setattr(clinicaltrials, "_curl_json", reject)
    with pytest.raises(ProviderRequestRejected):
        clinicaltrials.search("x")


# ---------- bioRxiv licence ----------

def test_biorxiv_lookup_exposes_licence_but_not_abstract(tmp_path, monkeypatch):
    monkeypatch.setattr(biorxiv, "web_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(biorxiv.time, "sleep", lambda *_: None)
    monkeypatch.setattr(biorxiv, "_curl_json", lambda url, retries=3: {"collection": [{
        "server": "biorxiv", "title": "T", "version": "2", "date": "2025-01-01",
        "category": "genomics", "type": "new results", "published": "NA",
        "license": "cc_no", "abstract": "never re-exposed",
    }]})
    rec = biorxiv.lookup("10.1101/2025.01.01.1")
    assert rec["license"] == "cc_no"
    assert "never re-exposed" not in json.dumps(rec)


# ---------- transport wiring ----------

def test_search_transports_opt_into_fail_fast_400(monkeypatch):
    seen = {}

    def fake(url, **kw):
        seen[kw["provider"]] = kw.get("reject_400")
        return {}
    monkeypatch.setattr(europepmc, "curl_json", fake)
    monkeypatch.setattr(clinicaltrials, "curl_json", fake)
    europepmc._curl_json("https://x.test")
    clinicaltrials._curl_json("https://x.test")
    monkeypatch.setattr(openfda, "curl_json", fake)
    openfda._curl_json("https://x.test")
    assert seen == {"europepmc": True, "clinicaltrials": True, "openfda": True}
    assert _http.curl_json.__kwdefaults__["reject_400"] is False


# ---------- openFDA ----------

#: Trimmed from real responses (probed 2026-11-04). The label keeps one long
#: section the parser must drop, so the "no label prose on disk" test has teeth.
LABEL = {
    "set_id": "68cd4f06-70e1-40d8-bedb-609ec0afa471",
    "id": "a1b2c3",
    "effective_time": "20260616",
    "boxed_warning": ["WARNING: SERIOUS LIVER INJURY"],
    "indications_and_usage": ["1 INDICATIONS AND USAGE  ZOLGENSMA is an adeno-associated "
                              "virus vector-based gene therapy\n indicated for SMA."],
    "clinical_studies": ["LONG-CLINICAL-STUDIES-SECTION"],
    "openfda": {"application_number": ["BLA125694", "ANDA061905"],
                "brand_name": ["Zolgensma"],
                "generic_name": ["ONASEMNOGENE ABEPARVOVEC-XIOI"],
                "manufacturer_name": ["Novartis Gene Therapies, Inc."],
                "product_type": ["HUMAN PRESCRIPTION DRUG"],
                "route": ["INTRAVENOUS"],
                "pharm_class_epc": ["Adeno-associated Virus Vector-based Gene Therapy [EPC]"]},
}
PMA = {
    "pma_number": "P170019", "supplement_number": "", "applicant": "Foundation Medicine, Inc.",
    "trade_name": "FoundationOne CDx",
    "generic_name": "Next generation sequencing oncology panel",
    "product_code": "PQP", "advisory_committee_description": "Pathology",
    "expedited_review_flag": "Y", "decision_date": "2017-11-30", "decision_code": "APPR",
    "ao_statement": "Approval for the FoundationOne CDx (F1CDx).  This device is NGS-based.",
    "street_1": "400 Summer St.", "openfda": {"device_class": "3"},
}
K510 = {
    "k_number": "DEN210014", "device_name": "EaseVRx", "applicant": "Appliedvr, Inc.",
    "decision_date": "2021-11-16", "decision_code": "DENG", "decision_description": "Unknown",
    "clearance_type": "Direct", "advisory_committee_description": "Neurology",
    "product_code": "QRA", "contact": "a person", "openfda": {"device_class": "2"},
}


def test_label_parse_is_the_declared_schema_without_long_sections():
    rec = openfda.parse_label(LABEL)
    assert set(rec) == set(openfda.LABEL_FIELDS)
    assert rec["application_numbers"] == ["BLA125694"]  # an ANDA is never a lead's identity
    assert rec["effective_date"] == "2026-06-16"
    assert rec["has_boxed_warning"] is True
    assert rec["indications"].startswith("1 INDICATIONS AND USAGE ZOLGENSMA is an")
    assert "\n" not in rec["indications"]
    assert rec["url"] == ("https://dailymed.nlm.nih.gov/dailymed/lookup.cfm"
                          "?setid=68cd4f06-70e1-40d8-bedb-609ec0afa471")
    assert "LONG-CLINICAL-STUDIES-SECTION" not in json.dumps(rec)


def test_device_parses_are_their_declared_schemas():
    pma = openfda.parse_pma(PMA)
    assert set(pma) == set(openfda.PMA_FIELDS)
    assert pma["expedited_review"] is True and pma["device_class"] == "3"
    assert pma["ao_statement"] == "Approval for the FoundationOne CDx (F1CDx). This device is NGS-based."
    assert pma["url"].endswith("pma.cfm?id=P170019")
    assert "Summer" not in json.dumps(pma)
    k = openfda.parse_510k(K510)
    assert set(k) == set(openfda.K510_FIELDS)
    assert k["is_de_novo"] is True and k["url"].endswith("denovo.cfm?ID=DEN210014")
    assert "a person" not in json.dumps(k)
    assert openfda.parse_510k({**K510, "k_number": "K260235"})["url"].endswith("pmn.cfm?ID=K260235")


def test_cp1252_artefacts_in_names_are_repaired():
    """openFDA stores `™` as U+0099; a terminal would print a control character."""
    assert openfda.parse_510k({**K510, "device_name": "Oncomine \x99 Dx"})["device_name"] == "Oncomine ™ Dx"
    assert openfda.parse_pma({**PMA, "trade_name": "Signatera\x99 CDx\x81"})["trade_name"] == "Signatera™ CDx"


@pytest.mark.parametrize(("query", "expected"), [
    ("prime editing", '("prime" AND "editing")'),
    ('"gene therapy" sickle', '("gene therapy" AND "sickle")'),
    ("car-t OR tcr", '("car-t" OR "tcr")'),
    ('(car-t OR "chimeric antigen receptor") persistence',
     '(( "car-t" OR "chimeric antigen receptor" ) AND "persistence")'),
    ('indications_and_usage:"sickle cell"', '(indications_and_usage:"sickle cell")'),
    ("openfda.brand_name:Kymriah", "(openfda.brand_name:Kymriah)"),
    ("Kymriah: tisagenlecleucel", '("Kymriah:" AND "tisagenlecleucel")'),
])
def test_openfda_query_building(query, expected):
    assert openfda.build_query(query) == expected


def test_label_words_are_scoped_to_what_the_product_is_for():
    """A bare word matches anywhere in ~110 KB of label prose; scope it."""
    q = openfda.build_query("aav OR crispr", fields=("a", "b"))
    assert q == '((a:"aav" OR b:"aav") OR (a:"crispr" OR b:"crispr"))'
    assert openfda.build_query('indications_and_usage:"x" y', fields=("a",)) == \
        '(indications_and_usage:"x" y)'  # field syntax: the user scoped it already


def test_openfda_filters_and_window_stay_outside_the_users_or():
    import datetime as dt
    label = openfda.build_query("a OR b", filters=openfda.LABEL_FILTER, date_field="effective_time",
                                since=dt.date(2025, 1, 2), today=dt.date(2026, 11, 4))
    assert label == ('("a" OR "b") AND openfda.application_number:(BLA* OR NDA*) '
                     "AND effective_time:[20250102 TO 20261104]")
    device = openfda.build_query("ngs", filters=openfda.PMA_FILTER, date_field="decision_date",
                                 since=dt.date(2025, 1, 2), today=dt.date(2026, 11, 4))
    assert device == '("ngs") AND supplement_number:"" AND decision_date:[2025-01-02 TO 2026-11-04]'


def test_openfda_search_caches_parsed_records_and_hides_the_key(cache, monkeypatch):
    monkeypatch.setenv("OPENFDA_API_KEY", "sekrit-fda")
    seen = []

    def fake(url, *, secret_query=None, retries=3):
        seen.append((url, secret_query))
        return {"meta": {}, "results": [LABEL]}

    monkeypatch.setattr(openfda, "_curl_json", fake)
    [rec] = openfda.search_labels("gene therapy", limit=5)
    assert rec["set_id"] == LABEL["set_id"]
    [(url, secret)] = seen
    assert "sekrit" not in url and secret == {"api_key": "sekrit-fda"}
    assert "limit=5" in url and "drug/label.json" in url
    assert "sort=effective_time%3Adesc" in url  # openFDA has no relevance order
    assert "indications_and_usage%3A%22gene%22" in url
    on_disk = "".join(p.read_text(encoding="utf-8") for p in cache.rglob("openfda__*.json"))
    assert on_disk and "LONG-CLINICAL-STUDIES-SECTION" not in on_disk
    assert not any("sekrit" in p.name for p in cache.rglob("*"))
    assert openfda.search_labels("gene therapy", limit=5) == [rec]  # served from cache
    assert len(seen) == 1


def test_openfda_no_match_404_is_zero_results(cache, monkeypatch):
    monkeypatch.setattr(openfda, "_curl_json", lambda url, **kw: {})
    assert openfda.search_pma("nothing") == []
    assert openfda.search_510k("nothing") == []


@pytest.mark.parametrize(("status", "body", "rejected"), [
    ("500", json.dumps({"error": {"code": "SERVER_ERROR",
                                  "details": "[parse_exception] parse_exception: EOF"}}), True),
    ("500", json.dumps({"error": {"code": "SERVER_ERROR", "message": "overloaded"}}), False),
    ("500", "<html>gateway</html>", False),
    ("503", json.dumps({"error": {"details": "parse_exception"}}), False),
])
def test_openfda_parse_error_is_a_rejected_query(status, body, rejected):
    assert openfda._parse_error(status, body) is rejected
