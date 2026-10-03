"""Download open-access PDFs for `scout search` leads into `inbox/`.

✅ Use when: a lead is worth ingesting and its PDF is openly available —
   arXiv, bioRxiv/medRxiv, or an open-access PubMed Central article.
❌ Don't use: for anything behind a paywall, for trial documents, or as a web
   fetch for prose. The only output is a PDF in `inbox/`, which then goes
   through the normal ingest path and its gates; that PDF is what Rule 3 means
   by "the PDF we have".

**The open-access decision is made from structured metadata, never guessed.**
arXiv and bioRxiv/medRxiv are free to read by construction (the licence is
recorded either way — `cc_no` means read-only, not unavailable). Anything else
needs Europe PMC's `isOpenAccess: Y` plus an OA PDF URL; without both, the key
is refused rather than probed.

**Downloads are confined to an allowlist of hosts** (`ALLOWED_HOSTS`), checked
on every redirect hop by `_http.curl_download`. A host that answers 403 or a
bot challenge produces a `manual` entry carrying the URL: a person in a
browser can usually still get it, and nothing is saved that is not a PDF.

**Loops stop at the first outage** (`errors.py` rule 3): every download goes
to a handful of hosts, so a failing network fails them all. What already
landed is reported, with `stopped_on` naming the key that broke.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..errors import EnvironmentFailure
from ..log import log
from ..paths import inbox_dir
from ..providers import biorxiv, europepmc
from ..providers._http import DownloadRefused, curl_download
from ..wiki import read_pages
from .recent import DeclineKeyError, normalize_key
from .search import wiki_doi_aliases

LOG_TAG = "scout-fetch"
MAX_BYTES = 100 * 1024 * 1024
ALLOWED_HOSTS = frozenset({
    "export.arxiv.org", "arxiv.org",
    "www.biorxiv.org", "www.medrxiv.org",
    "europepmc.org", "www.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
})
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.(.+)")
_PREPRINT_PREFIXES = ("10.1101/", "10.64898/")
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class Plan:
    key: str
    url: str | None = None
    filename: str | None = None
    doi: str | None = None            # passed to `agent ingest --doi`
    license: str | None = None
    refused: str | None = None        # reason the key cannot be fetched
    #: Europe PMC's `?pdf=render` links answered HTTP 403 to every non-browser
    #: client probed (2026-10-03). The attempt is still made; a dry run says
    #: up front that it will probably end as a manual download.
    best_effort: bool = False


def _slug(value: str) -> str:
    return _UNSAFE_NAME_RE.sub("-", value).strip("-.")


def arxiv_ingest_doi(arxiv_id: str) -> str:
    """`10.48550/arXiv.<id>`, capital X. `promote` writes `arxiv_id:` only for
    that spelling, and a `--doi` override is used verbatim."""
    return f"10.48550/arXiv.{arxiv_id}"


def _plan_arxiv(key: str, arxiv_id: str) -> Plan:
    return Plan(key=key, url=f"https://export.arxiv.org/pdf/{arxiv_id}",
                filename=f"arxiv-{_slug(arxiv_id)}.pdf", doi=arxiv_ingest_doi(arxiv_id),
                license="arxiv")


def _plan_preprint(key: str) -> Plan:
    record = biorxiv.lookup(key)
    if not record.get("server"):
        return Plan(key=key, refused="not found on bioRxiv or medRxiv")
    server, version = record["server"], record.get("version") or "1"
    return Plan(
        key=key,
        url=f"https://www.{server}.org/content/{key}v{version}.full.pdf",
        filename=f"{server}-{_slug(key.split('/', 1)[1])}v{version}.pdf",
        doi=key, license=record.get("license") or None,
    )


def _plan_europepmc(key: str) -> Plan:
    oa = europepmc.oa_status(key)
    if oa["is_open_access"] is None:
        return Plan(key=key, refused="no Europe PMC record")
    if not oa["is_open_access"] or not oa["pdf_urls"]:
        return Plan(key=key, refused="not open access per Europe PMC")
    pmcid = oa.get("pmcid")
    name = f"{pmcid}.pdf" if pmcid else f"{_slug(key)}.pdf"
    return Plan(key=key, url=oa["pdf_urls"][0], filename=name,
                doi=oa.get("doi"), license=oa.get("license"), best_effort=True)


def plan(raw_key: str) -> Plan:
    """Where `raw_key`'s open-access PDF would come from, or why it can't."""
    try:
        key = normalize_key(raw_key)
    except DeclineKeyError as exc:
        return Plan(key=raw_key, refused=str(exc))
    if key.startswith("nct:"):
        return Plan(key=key, refused="trial registrations have no paper PDF; "
                                     "trial documents are not fetched")
    if key.startswith("s2:"):
        return Plan(key=key, refused="Semantic Scholar ids carry no open-access link here")
    m = _ARXIV_DOI_RE.fullmatch(key)
    if m:
        return _plan_arxiv(key, m.group(1))
    if key.startswith(_PREPRINT_PREFIXES):
        return _plan_preprint(key)
    return _plan_europepmc(key)


def _ingest_command(path: str, doi: str | None) -> str:
    cmd = f"researchwiki agent ingest {path}"
    return f"{cmd} --doi {doi}" if doi else cmd


def fetch(keys: list[str], *, dry_run: bool = False) -> dict:
    """Plan and (unless `dry_run`) download each key. Never raises for one key.

    Report shape: `{fetched, already_present, skipped, manual, stopped_on,
    error, dry_run}` — the `--json` contract. `fetched` / `already_present`
    entries are `{key, path, url, license, best_effort, ingest_doi,
    ingest_command}`.
    """
    report: dict = {"fetched": [], "already_present": [], "skipped": [], "manual": [],
                    "stopped_on": None, "error": None, "dry_run": dry_run}
    wiki_dois = wiki_doi_aliases(read_pages())
    inbox = inbox_dir()
    for raw in keys:
        try:
            p = plan(raw)
            if p.refused:
                report["skipped"].append({"key": p.key, "reason": p.refused})
                continue
            if {p.key, (p.doi or "").lower()} & wiki_dois:
                report["skipped"].append({"key": p.key, "reason": "already in the wiki"})
                continue
            dest = inbox / p.filename
            rel = f"inbox/{p.filename}"
            entry = {"key": p.key, "path": rel, "url": p.url, "license": p.license,
                     "best_effort": p.best_effort, "ingest_doi": p.doi,
                     "ingest_command": _ingest_command(rel, p.doi)}
            if dest.exists():
                report["already_present"].append(entry)
                continue
            if dry_run:
                report["fetched"].append(entry)
                continue
            inbox.mkdir(parents=True, exist_ok=True)
            curl_download(p.url, dest, provider=LOG_TAG, allowed_hosts=ALLOWED_HOSTS,
                          max_bytes=MAX_BYTES)
            log(f"saved {rel}", tag=LOG_TAG)
            report["fetched"].append(entry)
        except DownloadRefused as exc:
            report["manual"].append({"key": p.key, "url": exc.url, "reason": exc.reason})
        except FileExistsError:
            report["already_present"].append(entry)
        except EnvironmentFailure as exc:
            report["stopped_on"] = raw
            report["error"] = str(exc)
            log(f"stopping: {exc}", tag=LOG_TAG)
            break
    return report
