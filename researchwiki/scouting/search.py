"""Keyword literature search across PubMed, arXiv, bioRxiv/medRxiv, ClinicalTrials.gov.

✅ Use when: "is there anything on <topic> I don't have?" — a query, not a
   seed set. Complements `scout recent` (S2 recommendations near existing
   pages) and `neighbors` (the citation graph around one paper).
❌ Don't use: as evidence. Every result is a **discovery-only lead**; nothing
   here may support wiki prose, a claim, or a `[[wikilink]]` until the PDF is
   ingested. `scout search fetch` (`search_fetch.py`) is the bridge.

**Rule 1.** Searching is discovery, not grounding: Rule 1 governs what may
*support* a claim, and nothing here does. Each record keeps the source's
verbatim abstract (papers) or brief summary (trials) because that is what a
person reads to decide whether to ingest; it is embedded locally for ranking,
stored in the snapshot and `--json`, and printed in the terminal on
`--abstracts`. It never reaches page authoring, the claims DB, or a model
prompt — that takes the ingested PDF.

**Sources fail independently.** `errors.py` rule 3 says a loop over items
stops at the first `EnvironmentFailure`, because failures across items are
correlated. These source queries run separately, including one Europe PMC
query per preprint server, so an unavailable source is recorded and the rest
still answer. The CLI exits 2 after printing them.

**Ranking reuses `scout recent`'s scorer** (`recent.score_candidates`): fit to
the nearest wiki paper pages, and matches against synthesis/idea "What would
update this page" triggers and open proposals. Papers and trials are scored in
separate calls so trials cannot take a paper's per-trigger slot.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from dataclasses import dataclass, field

from ..fsatomic import read_json, write_json_atomic
from ..log import log
from ..metadata_sanity import UNUSABLE_SURNAMES, YEAR_TOLERANCE, is_preprint_doi
from ..paths import web_cache_dir
from ..providers import arxiv, clinicaltrials, europepmc, pubmed
from ..providers._http import ProviderRequestRejected, StructuredProviderUnavailable
from ..stems import strip_diacritics
from ..wiki import Page, read_pages
from . import recent as R

LOG_TAG = "scout-search"
SCHEMA_VERSION = 1
SOURCES = ("pubmed", "arxiv", "biorxiv", "medrxiv", "clinicaltrials")
DEFAULT_SOURCES = ("pubmed", "arxiv", "biorxiv", "medrxiv")
DEFAULT_LIMIT = 20
MAX_LIMIT = 100
PREPRINT_DOI_PREFIXES = ("10.1101/", "10.64898/")
_PREPRINT_SOURCES = frozenset({"arxiv", "biorxiv", "medrxiv"})
_NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


# ---------- same work, by title ----------

#: PubMed's initials block: `AV`, `A V`, `A.V.` — never a surname.
_INITIALS_RE = re.compile(r"(?:[A-Z]\.?\s*){1,4}")
_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})


def surname(name: str) -> str:
    """The first author's surname, normalized, from either name order.

    PubMed writes `Smith AV` (surname, then an initials block); arXiv,
    Europe PMC and wiki pages write `Andrew V. Smith` or `Andrew V Smith`. The
    initials block is recognised by shape — capitals only, at most four,
    optionally dotted — so `Smith AV` gives `smith` and `Andrew V Smith` gives
    `smith`. A multi-word surname compares on its last word in both orders
    (`van der Berg JM` and `Jan M van der Berg` both give `berg`); a
    hyphenated one stays whole. Comparing every name token instead let a shared `av` (or a shared
    given name) pass two different people as one. A mononym (`DeepSeek-AI`) is
    its own surname; `et al.`, `Jr` and an empty name give "".
    """
    toks = [t for t in (name or "").replace(",", " ").split()
            if t.lower().strip(".") not in _NAME_SUFFIXES]
    if not toks:
        return ""
    if len(toks) >= 2 and _INITIALS_RE.fullmatch(toks[-1]):
        raw = toks[-2]             # `Smith AV`; `van der Berg JM` -> `Berg`
    elif toks[-1].lower().strip(".") == "al" and len(toks) >= 3:
        raw = toks[-3]             # `Guohui Chuai et al.`
    else:
        raw = toks[-1]
    norm = re.sub(r"[^a-z0-9]", "", strip_diacritics(raw).lower())
    return "" if norm in UNUSABLE_SURNAMES else norm


@dataclass(frozen=True)
class WorkSig:
    """What a title-only match must agree on before it counts as one work."""

    dois: frozenset[str]
    first_author: str               # normalized surname; "" when unknown
    year: int | None
    preprint: bool
    pmids: frozenset[str] = frozenset()


def same_work_by_title(a: WorkSig, b: WorkSig) -> bool:
    """Whether two records sharing a normalized title describe one work.

    A title is not an identity — "Editorial", "Correction", "Reply" and many
    real titles recur across unrelated papers — so a title match must be
    corroborated by the first author's surname and is refused on any
    contradiction:

    - the surnames differ, or either is unknown;
    - both sides have a PMID and they differ — PubMed indexes a work once;
    - both sides carry a DOI of the same kind (preprint or published) and
      they differ: a preprint and its journal version legitimately have two
      DOIs, two journal articles with one title are two papers;
    - the years differ by more than `YEAR_TOLERANCE` and neither side is a
      preprint (a journal version can follow its preprint by years).

    Wiki pages record no PMID, so against a page the surname and DOI carry it.
    """
    if not a.first_author or a.first_author != b.first_author:
        return False
    if a.pmids and b.pmids and not (a.pmids & b.pmids):
        return False
    if any(da != db and is_preprint_doi(da) == is_preprint_doi(db)
           for da in a.dois for db in b.dois):
        return False
    if a.year and b.year and not (a.preprint or b.preprint):
        return abs(a.year - b.year) <= YEAR_TOLERANCE
    return True


# ---------- leads ----------

@dataclass
class Lead:
    kind: str                         # "paper" | "trial"
    sources: list[str]
    ids: dict[str, str]               # doi, journal_doi, pmid, pmcid, arxiv_id, nct_id, server, version
    title: str
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    publication_date: str | None = None
    year: int | None = None
    retracted: bool = False
    trial: dict | None = None
    fit: float | None = None
    nearest: list[dict] = field(default_factory=list)
    triggers: list[dict] = field(default_factory=list)
    first_seen: str | None = None
    #: Verbatim abstract (papers). `score_text` is the embedding input and is
    #: not serialized; for a trial it is title + conditions + interventions.
    abstract: str = field(default="", repr=False)
    score_text: str = field(default="", repr=False)

    @property
    def key(self) -> str:
        """The one identifier `fetch` and `--decline` take for this lead."""
        ids = self.ids
        if ids.get("doi"):
            return ids["doi"]
        if ids.get("pmid"):
            return f"pmid:{ids['pmid']}"
        if ids.get("pmcid"):
            return f"pmcid:{ids['pmcid'].lower()}"
        return f"nct:{ids['nct_id'].lower()}"

    @property
    def fetch_key(self) -> str | None:
        """The key `fetch` should be given, when it differs in practice.

        A lead merged from PubMed and arXiv is keyed by its journal DOI, but
        the PDF that can actually be downloaded is the arXiv one.
        """
        ids = self.ids
        if self.kind == "trial":
            return None
        if ids.get("arxiv_id"):
            return f"10.48550/arxiv.{ids['arxiv_id'].lower()}"
        doi = ids.get("doi") or ""
        if doi.startswith(PREPRINT_DOI_PREFIXES):
            return doi
        if ids.get("pmcid"):
            return f"pmcid:{ids['pmcid'].lower()}"
        return self.key

    def alias_keys(self) -> set[str]:
        """Every key this lead could have been declined or ingested under."""
        ids = self.ids
        keys = {self.key}
        for name in ("doi", "journal_doi"):
            if ids.get(name):
                keys.add(ids[name])
        if ids.get("arxiv_id"):
            keys.add(f"10.48550/arxiv.{ids['arxiv_id'].lower()}")
        if ids.get("pmid"):
            keys.add(f"pmid:{ids['pmid']}")
        if ids.get("pmcid"):
            keys.add(f"pmcid:{ids['pmcid'].lower()}")
        if ids.get("nct_id"):
            keys.add(f"nct:{ids['nct_id'].lower()}")
        return keys

    def dois(self) -> set[str]:
        return {k for k in self.alias_keys() if k.startswith("10.")}

    def work_sig(self) -> WorkSig:
        dois = frozenset(self.dois())
        return WorkSig(
            dois=dois,
            first_author=surname(self.authors[0]) if self.authors else "",
            year=self.year,
            preprint=bool(set(self.sources) & _PREPRINT_SOURCES)
            or any(is_preprint_doi(d) for d in dois),
            pmids=frozenset({self.ids["pmid"]}) if self.ids.get("pmid") else frozenset(),
        )

    def to_dict(self) -> dict:
        out = {
            "key": self.key,
            "kind": self.kind,
            "sources": list(self.sources),
            "doi": self.ids.get("doi"),
            "ids": dict(self.ids),
            "fetch_key": self.fetch_key,
            "title": self.title,
            "authors": list(self.authors),
            "venue": self.venue,
            "publication_date": self.publication_date,
            "year": self.year,
            "retracted": self.retracted,
            "has_abstract": bool(self.abstract),
            "abstract": self.abstract,
            "fit": self.fit,
            "nearest": self.nearest,
            "triggers": self.triggers,
            "first_seen": self.first_seen,
        }
        if self.trial is not None:
            out["trial"] = self.trial
        return out


def _ids(**kw) -> dict[str, str]:
    return {k: str(v) for k, v in kw.items() if v}


def _year_of(value: str | None) -> int | None:
    return int(value[:4]) if value and value[:4].isdigit() else None


def from_pubmed(r: dict) -> Lead:
    return Lead(
        kind="paper", sources=["pubmed"],
        ids=_ids(doi=r.get("doi"), pmid=r.get("pmid"), pmcid=r.get("pmcid")),
        title=r.get("title") or "", authors=list(r.get("authors") or []),
        venue=r.get("journal") or "", publication_date=r.get("pub_date") or None,
        year=r.get("year"), retracted=bool(r.get("retracted")),
        abstract=r.get("abstract") or "",
    )


def from_arxiv(r: dict) -> Lead:
    return Lead(
        kind="paper", sources=["arxiv"],
        ids=_ids(doi=r.get("doi"), journal_doi=r.get("journal_doi"),
                 arxiv_id=r.get("arxiv_id"), version=r.get("version")),
        title=r.get("title") or "", authors=list(r.get("authors") or []),
        venue=f"arXiv ({r['primary_category']})" if r.get("primary_category") else "arXiv",
        publication_date=r.get("published") or None, year=r.get("year"),
        abstract=r.get("abstract") or "",
    )


def from_preprint(r: dict) -> Lead:
    server = r.get("server") or ""
    return Lead(
        kind="paper", sources=[server or "biorxiv"],
        ids=_ids(doi=r.get("doi"), server=server),
        title=r.get("title") or "", authors=list(r.get("authors") or []),
        venue={"biorxiv": "bioRxiv", "medrxiv": "medRxiv"}.get(server, server),
        publication_date=r.get("pub_date") or None, year=r.get("year"),
        abstract=r.get("abstract") or "",
    )


def from_trial(r: dict) -> Lead:
    trial = {k: r[k] for k in clinicaltrials.TRIAL_FIELDS
             if k not in ("nct_id", "brief_title", "brief_summary")}
    interventions = [i["name"] for i in r.get("interventions") or [] if i.get("name")]
    return Lead(
        kind="trial", sources=["clinicaltrials"], ids=_ids(nct_id=r.get("nct_id")),
        title=r.get("brief_title") or "", venue="ClinicalTrials.gov",
        publication_date=r.get("start_date") or None, year=_year_of(r.get("start_date")),
        trial=trial,
        abstract=r.get("brief_summary") or "",
        score_text="\n".join(x for x in (
            r.get("brief_title"), r.get("official_title"),
            "; ".join(r.get("conditions") or []), "; ".join(interventions),
        ) if x),
    )


# ---------- per-source runs ----------

def _run_source(source: str, query: str, *, limit: int,
                since: _dt.date | None, max_age_days: float) -> list[Lead]:
    if source == "pubmed":
        return [from_pubmed(r) for r in pubmed.search(query, limit=limit, since=since,
                                                      max_age_days=max_age_days)]
    if source == "arxiv":
        return [from_arxiv(r) for r in arxiv.search(query, limit=limit, since=since,
                                                    max_age_days=max_age_days)]
    if source in ("biorxiv", "medrxiv"):
        return [from_preprint(r) for r in europepmc.search_preprints(
            query, servers=[source], limit=limit, since=since, max_age_days=max_age_days)]
    return [from_trial(r) for r in clinicaltrials.search(query, limit=limit, since=since,
                                                         max_age_days=max_age_days)]


def gather(query: str, sources: list[str], *, limit: int, since: _dt.date | None,
           max_age_days: float) -> tuple[list[Lead], dict[str, dict]]:
    """Every requested source's leads, plus a per-source status map."""
    leads: list[Lead] = []
    status: dict[str, dict] = {}
    for source in sources:
        try:
            got = _run_source(source, query, limit=limit, since=since,
                              max_age_days=max_age_days)
        except ProviderRequestRejected as exc:
            status[source] = {"status": "rejected", "returned": 0, "error": str(exc)}
            log(f"WARN: {exc}", tag=LOG_TAG)
            continue
        except StructuredProviderUnavailable as exc:
            status[source] = {"status": "unavailable", "returned": 0, "error": str(exc)}
            log(f"WARN: {exc}", tag=LOG_TAG)
            continue
        status[source] = {"status": "ok", "returned": len(got), "error": None}
        leads.extend(got)
    return leads, status


# ---------- merge and filter ----------

def arxiv_id_text(value) -> str:
    """An `arxiv_id:` frontmatter value as the identifier it was written as.

    Unquoted, `arxiv_id: 2003.02320` parses as the float `2003.0232` and
    `0704.0001` as `704.0001`: YAML drops trailing zeros and, for 2007-2009
    ids, the leading one, and the alias would match nothing. New-style ids
    are fixed-width — a four-digit YYMM, then a sequence number of five digits
    from 1501 (January 2015) and four before — so both are recoverable. `promote`
    quotes the value for exactly this reason; older and hand-written pages
    don't always.
    """
    if isinstance(value, float):
        yymm, _, seq = repr(value).partition(".")
        # 0704.0001 parses as 704.0001: the year's leading zero goes too.
        yymm = yymm.zfill(4)
        width = 5 if yymm >= "1501" else 4
        return f"{yymm}.{seq.ljust(width, '0')}"
    return str(value or "").strip().lower()


def wiki_doi_aliases(pages: list[Page]) -> set[str]:
    """DOIs already held by the wiki, including a paper's retained arXiv ID.

    A preprint page may later switch its `doi` to the journal DOI while keeping
    `arxiv_id`. Both versions must still count as the same held paper.
    """
    dois = {d for p in pages if (d := R._doi(p))}
    for page in R._paper_pages(pages):
        arxiv_id = arxiv_id_text(page.fm.get("arxiv_id"))
        if arxiv_id:
            dois.add(f"10.48550/arxiv.{arxiv_id}")
    return dois


def page_work_sig(page: Page) -> WorkSig:
    dois = set()
    if (doi := R._doi(page)):
        dois.add(doi)
    if (arxiv_id := arxiv_id_text(page.fm.get("arxiv_id"))):
        dois.add(f"10.48550/arxiv.{arxiv_id}")
    authors = page.fm.get("authors")
    if isinstance(authors, list):
        first = str(authors[0]) if authors else ""
    else:
        first = re.split(r"[;,]", str(authors or ""), maxsplit=1)[0]
    return WorkSig(frozenset(dois), surname(first), R._year(page) or None,
                   any(is_preprint_doi(d) for d in dois))


def merge(leads: list[Lead]) -> list[Lead]:
    """One lead per work: a shared identifier, or a corroborated title.

    The first lead seen (source order) keeps its key and title; later ones add
    their sources and fill identifiers it lacked. A journal paper on PubMed and
    its arXiv preprint merge through the arXiv entry's `journal_doi`; without
    one, through the same normalized title *and* `same_work_by_title`. Trials
    always carry an NCT id, so they merge on it alone.
    """
    out: list[Lead] = []
    by_key: dict[str, Lead] = {}
    by_title: dict[str, list[Lead]] = {}
    for lead in leads:
        title = R._norm_title(lead.title) if lead.kind == "paper" else ""
        match = next((by_key[k] for k in lead.alias_keys() if k in by_key), None)
        if match is None and title:
            sig = lead.work_sig()
            match = next((m for m in by_title.get(title, ())
                          if same_work_by_title(sig, m.work_sig())), None)
        if match is None:
            out.append(lead)
            match = lead
        else:
            for s in lead.sources:
                if s not in match.sources:
                    match.sources.append(s)
            for k, v in lead.ids.items():
                match.ids.setdefault(k, v)
            if len(lead.abstract) > len(match.abstract):
                match.abstract = lead.abstract
            match.retracted = match.retracted or lead.retracted
            match.authors = match.authors or lead.authors
            match.publication_date = match.publication_date or lead.publication_date
            match.year = match.year or lead.year
        for k in match.alias_keys():
            by_key[k] = match
        if title:
            same = by_title.setdefault(title, [])
            if not any(m is match for m in same):
                same.append(match)
    return out


@dataclass
class WikiIndex:
    dois: set[str]
    titles: dict[str, list[WorkSig]]
    ncts: set[str]

    @classmethod
    def from_pages(cls, pages: list[Page]) -> "WikiIndex":
        dois = wiki_doi_aliases(pages)
        titles: dict[str, list[WorkSig]] = {}
        for p in R._paper_pages(pages):
            title = R._norm_title(str(p.fm.get("title") or ""))
            if title:
                titles.setdefault(title, []).append(page_work_sig(p))
        ncts = {m.lower() for p in pages
                for m in _NCT_RE.findall(p.str_field("document_id"))}
        return cls(dois, titles, ncts)

    def holds(self, lead: Lead) -> bool:
        """Held under a shared DOI, or a title the page's own record corroborates.

        The title path catches a preprint whose journal version the wiki holds
        under another DOI; `same_work_by_title` keeps it from also hiding an
        unrelated paper that happens to share the title.
        """
        if lead.kind == "trial":
            return (lead.ids.get("nct_id") or "").lower() in self.ncts
        if lead.dois() & self.dois:
            return True
        sig = lead.work_sig()
        return any(same_work_by_title(sig, page)
                   for page in self.titles.get(R._norm_title(lead.title), ()))


def filter_leads(leads: list[Lead], *, wiki: WikiIndex, declined: set[str],
                 cutoff: _dt.date | None) -> tuple[list[Lead], dict[str, int]]:
    counts = {"in_wiki": 0, "declined": 0, "outside_window": 0}
    out: list[Lead] = []
    for lead in leads:
        if wiki.holds(lead):
            counts["in_wiki"] += 1
            continue
        if lead.alias_keys() & declined:
            counts["declined"] += 1
            continue
        if cutoff is not None:
            published = R._parse_date(lead.publication_date)
            if published is not None and published < cutoff:
                counts["outside_window"] += 1
                continue
        if not lead.score_text:
            lead.score_text = "\n\n".join(x for x in (lead.title, lead.abstract) if x)
        out.append(lead)
    return out, counts


# ---------- orchestration ----------

def _search_dir():
    return web_cache_dir() / "search"


def mark_first_seen(leads: list[Lead], today: _dt.date) -> None:
    """Same ledger idea as `recent.mark_first_seen`, kept separate so a lead
    seen by one command is still NEW the first time the other shows it."""
    path = _search_dir() / "seen.json"
    seen = read_json(path, {})
    if not isinstance(seen, dict):
        seen = {}
    for lead in leads:
        lead.first_seen = seen.setdefault(lead.key, today.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, seen)


def run(
    query: str,
    *,
    sources: list[str] = DEFAULT_SOURCES,
    limit: int = DEFAULT_LIMIT,
    since: _dt.date | None = None,
    max_age_days: float = 1,
    today: _dt.date | None = None,
) -> dict:
    """The snapshot, also written to `.web-cache/search/runs/`."""
    today = today or _dt.date.today()
    raw, status = gather(query, list(sources), limit=limit, since=since,
                         max_age_days=max_age_days)
    merged = merge(raw)
    pages = read_pages()
    declines = R.load_declines()
    leads, counts = filter_leads(merged, wiki=WikiIndex.from_pages(pages),
                                 declined=set(declines), cutoff=since)
    papers = [x for x in leads if x.kind == "paper"]
    trials = [x for x in leads if x.kind == "trial"]
    triggers = R.collect_triggers(pages)
    scored_papers = R.score_candidates(papers, triggers)
    scored = R.score_candidates(trials, triggers) and scored_papers
    if leads and not scored:
        log("WARN: page index or embedding model unavailable — leads are unranked "
            "and unmatched; run `researchwiki reindex`", tag=LOG_TAG)
    mark_first_seen(leads, today)
    updates, rest = R.rank(papers)
    trial_updates, trial_rest = R.rank(trials)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
        "query": query,
        "sources_requested": list(sources),
        "limit": limit,
        "since": since.isoformat() if since else None,
        "sources": status,
        "counts": {"returned": len(raw), "merged": len(merged), **counts,
                   "papers": len(papers), "trials": len(trials)},
        "scored": scored,
        "triggers_indexed": len(triggers),
        "page_matches": [x.to_dict() for x in updates],
        "leads": [x.to_dict() for x in rest],
        "trials": [x.to_dict() for x in trial_updates + trial_rest],
    }
    digest = hashlib.md5(
        "|".join([query, ",".join(sources), snapshot["since"] or ""]).encode()
    ).hexdigest()[:8]
    out = _search_dir() / "runs" / f"{today.isoformat()}__{digest}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, snapshot)
    snapshot["snapshot_path"] = str(out)
    return snapshot
