"""`scout search` providers: fixed schemas and error shapes.

Each parser is fed an inline payload shaped like the real API's (probed
2026-10-03) and must emit exactly its declared fields, so a provider change
that adds or drops a key is a visible, reviewed edit. Hermetic — transports
are faked.
"""

from __future__ import annotations

import json

import pytest

from researchwiki.providers import _http, arxiv, biorxiv, clinicaltrials, europepmc, pubmed
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
    for mod in (pubmed, arxiv, europepmc, clinicaltrials):
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
    "authorList": {"author": [{"fullName": "Weller J"}]},
    "bookOrReportDetails": {"publisher": "bioRxiv"},
    "isOpenAccess": "N", "citedByCount": 3,
}]}}


def test_preprint_search_schema_and_query(cache, monkeypatch):
    urls = []
    monkeypatch.setattr(europepmc, "_curl_json", lambda url, retries=3: urls.append(url) or EPMC_PREPRINT)
    [rec] = europepmc.search_preprints("long deletions", servers=["biorxiv", "medrxiv"], limit=5)
    assert set(rec) == set(europepmc.SEARCH_FIELDS)
    assert rec["server"] == "biorxiv" and rec["title"] == "Generating long deletions"
    assert "SRC%3APPR" in urls[0] and "medRxiv" in urls[0]


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
    assert seen == {"europepmc": True, "clinicaltrials": True}
    assert _http.curl_json.__kwdefaults__["reject_400"] is False
