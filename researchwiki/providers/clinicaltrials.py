"""ClinicalTrials.gov API v2 thin client — trial registrations for `scout search`.

A trial registration is not a paper, so this returns a separate fixed-schema
*trial* record: registry identity, status, design, sponsor, dates,
enrollment, linked publication PMIDs, whether a protocol / statistical
analysis plan is posted, and the registry's brief summary — what a person
reads to judge whether the trial matters. The longer registry prose
(detailed description, eligibility criteria, outcome descriptions, citation
strings) is not requested: it is long and adds little to triage.

API docs: https://clinicaltrials.gov/data-api/api. Probed 2026-10-03: an
unknown field name is HTTP 400 (`LargeDocHasSap` is wrong; `LargeDocHasSAP`
is right, though the response key is `hasSap`); posted documents live at
`https://cdn.clinicaltrials.gov/large-docs/<last 2 digits>/<NCT>/<filename>`.
"""

from __future__ import annotations

import time
import urllib.parse
from datetime import date

from ..paths import web_cache_dir
from ._cache import read_cache, safe_cache_key, write_stamped_cache
from ._http import curl_json

STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
DOCS_BASE = "https://cdn.clinicaltrials.gov/large-docs"
POLITE_SLEEP = 0.3
#: Requested fields, as API field names. `parse_study` reads exactly these.
API_FIELDS = (
    "NCTId", "BriefTitle", "OfficialTitle", "BriefSummary", "OverallStatus", "Phase", "StudyType",
    "Condition", "InterventionName", "InterventionType", "LeadSponsorName",
    "StartDate", "PrimaryCompletionDate", "CompletionDate", "EnrollmentCount",
    "HasResults", "ReferencePMID", "ReferenceType", "LargeDocLabel",
    "LargeDocHasProtocol", "LargeDocHasSAP", "LargeDocFilename",
)
TRIAL_FIELDS = (
    "nct_id", "brief_title", "official_title", "brief_summary", "overall_status", "phases",
    "study_type", "conditions", "interventions", "lead_sponsor", "start_date",
    "primary_completion_date", "completion_date", "enrollment", "has_results",
    "references", "documents",
)


def _curl_json(url: str, retries: int = 3) -> dict:
    return curl_json(url, provider="clinicaltrials", retries=retries, reject_400=True)


def _date(module: dict, key: str) -> str:
    return str((module.get(key) or {}).get("date") or "")


def doc_url(nct_id: str, filename: str) -> str:
    return f"{DOCS_BASE}/{nct_id[-2:]}/{nct_id}/{filename}"


def parse_study(study: dict) -> dict:
    """One `studies[]` element → a `TRIAL_FIELDS` record."""
    proto = study.get("protocolSection") or {}
    ident = proto.get("identificationModule") or {}
    status = proto.get("statusModule") or {}
    design = proto.get("designModule") or {}
    nct = str(ident.get("nctId") or "")
    docs = ((study.get("documentSection") or {}).get("largeDocumentModule") or {}).get("largeDocs") or []
    return {
        "nct_id": nct,
        "brief_title": str(ident.get("briefTitle") or ""),
        "official_title": str(ident.get("officialTitle") or ""),
        "brief_summary": str((proto.get("descriptionModule") or {}).get("briefSummary") or ""),
        "overall_status": str(status.get("overallStatus") or ""),
        "phases": [str(p) for p in design.get("phases") or []],
        "study_type": str(design.get("studyType") or ""),
        "conditions": [str(c) for c in (proto.get("conditionsModule") or {}).get("conditions") or []],
        "interventions": [
            {"name": str(i.get("name") or ""), "type": str(i.get("type") or "")}
            for i in (proto.get("armsInterventionsModule") or {}).get("interventions") or []
        ],
        "lead_sponsor": str(((proto.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}).get("name") or ""),
        "start_date": _date(status, "startDateStruct"),
        "primary_completion_date": _date(status, "primaryCompletionDateStruct"),
        "completion_date": _date(status, "completionDateStruct"),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "has_results": bool(study.get("hasResults")),
        "references": [
            {"pmid": str(r.get("pmid")), "type": str(r.get("type") or "")}
            for r in (proto.get("referencesModule") or {}).get("references") or []
            if r.get("pmid")
        ],
        "documents": [
            {"label": str(d.get("label") or ""), "has_protocol": bool(d.get("hasProtocol")),
             "has_sap": bool(d.get("hasSap")),
             "url": doc_url(nct, str(d["filename"])) if d.get("filename") and nct else None}
            for d in docs
        ],
    }


def search(query: str, *, limit: int = 20, since: date | None = None,
           max_age_days: float = 1) -> list[dict]:
    """Trials matching `query`, in the API's relevance order.

    `since` keeps trials whose start date is on or after it.
    """
    params = {"query.term": query, "pageSize": str(limit), "fields": ",".join(API_FIELDS),
              "format": "json"}
    if since:
        params["filter.advanced"] = f"AREA[StartDate]RANGE[{since.isoformat()},MAX]"
    url = f"{STUDIES_URL}?{urllib.parse.urlencode(params)}"
    cache_dir = web_cache_dir() / "search"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"clinicaltrials__{safe_cache_key(url)}.json"
    data = read_cache(cache, max_age_days=max_age_days)
    if data is None:
        data = _curl_json(url)
        write_stamped_cache(cache, data)
        time.sleep(POLITE_SLEEP)
    return [parse_study(s) for s in data.get("studies") or []]
