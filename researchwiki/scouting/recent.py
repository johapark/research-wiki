"""Recently published papers near the corpus, via S2 multi-seed recommendations.

✅ Use when: asking "what has come out lately that this wiki should know
   about?" for a category, a synthesis or idea page, or a hand-picked set of
   papers.
❌ Don't use: for the graph around one paper (`neighbors`) or for gaps the
   whole corpus cites (`scout citations`). Both walk citation edges, and a
   paper published last month has none yet — that is the gap this fills.

**Rule 1.** Only whitelisted Semantic Scholar fields leave the provider: ids,
title, publication date, venue, citation count, and the verbatim abstract. The
abstract is embedded locally to score the candidate against the wiki. Like every
S2 response it sits in the raw `.s2-cache/` file; it never reaches the
snapshot, the terminal, or a model prompt, and no model is called at all.
Every candidate is a lead: nothing here can support wiki prose, a claim, or a
`[[wikilink]]` until its PDF is ingested.

**Ranking is local.** S2 returns up to 100 recent papers similar to the seeds
(and unlike any declined ones). Each is embedded with the page index's own
bi-encoder, then scored two ways:

* `fit` — mean cosine to its three nearest wiki paper pages;
* `triggers` — each item under a synthesis or idea page's "What would
  update this page" / "What would change the conclusion" heading, and the
  decisive uncertainty of every open proposal. A candidate matches a trigger
  only when both hold:

  - its cosine to the trigger is at least `TRIGGER_Z_MIN` standard
    deviations above that trigger's mean cosine to the wiki's own paper
    pages. A raw threshold does not work: a generically worded trigger sits
    near everything, and at a fixed 0.80 one CRISPR off-target bullet matched
    ten of twelve single-cell candidates. Standardizing per trigger removes
    those hubs.
  - at least `MIN_NEIGHBOR_OVERLAP` of its `NEIGHBOR_K` nearest wiki papers
    are cited by the trigger's page — the paper lands in the part of the
    corpus that page is about, not just near its wording.
  - its mean cosine to the three papers that page cites nearest to it is at
    least `PAGE_RELEVANCE_MIN`. This is the gate that carries the precision:
    on blind labels it separated on-topic from off-topic pairs far better
    than any trigger-text score did (AUC 0.90 vs 0.61).

  Each trigger then keeps only its `PER_TRIGGER_CAP` best candidates in the
  run. A pool seeded from one field shares that field's vocabulary, so a
  broadly worded trigger can clear both gates for most of it (one FDR trigger
  did for 22 of 87 statistics candidates); the cap keeps the few that sit
  closest.

  A match says the paper is on that page's topic and names the trigger it
  sits closest to. Like a semantic-KNN cross-link candidate, it nominates and
  verifies nothing: whether the paper actually answers the trigger is for the
  reader to judge.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..categories import PAGE_TYPE_DIRS, content_categories
from ..fsatomic import read_json, write_json_atomic
from ..log import log
from ..paths import s2_cache_dir, wiki_root
from ..providers import ScholarlyArticle
from ..providers.semantic_scholar import MAX_RECOMMENDATION_SEEDS, SemanticScholarProvider
from ..wiki import Page, read_pages

LOG_TAG = "scout-recent"
SCHEMA_VERSION = 1
DEFAULT_DAYS = 60
DEFAULT_LIMIT = 20
#: Candidates requested from S2 per run. The endpoint's own ceiling is 500,
#: but past ~100 its recency weighting fades (a 36-seed probe at 500 reached
#: back 21 months), which defeats the point of a new-paper feed.
POOL = 100
#: Declined papers sent as negative seeds. They share S2's 100-id budget with
#: the positives, so keep the share small.
MAX_NEGATIVES = 20
FIT_TOP_K = 3
MIN_TRIGGER_WORDS = 6
#: Trigger-match gates. The first three were set by eye on four pools; the
#: relevance floor was then tuned against 614 blind-labelled (paper, page)
#: pairs from eight pools of 2026-09-27, chosen on four (single-cell,
#: statistics, genomics, one synthesis page's citations) and confirmed on four
#: held out (cgt, compbio, ai, genetics). Labeller agreement on a 59-pair
#: repeat: 97%, kappa 0.93. Adding the floor raised the share of matches
#: labelled on-topic from 0.67 to 0.90 on the tuning pools and from 0.70 to
#: 0.88 on the held-out ones.
#:
#: What it does not fix: the trigger text itself barely ranks. Among on-topic
#: papers, the ones that actually answer a trigger score no higher than the
#: rest (AUC 0.56-0.66), so read a match as "on this page's topic", with the
#: trigger shown as the likeliest reason, not as a confirmed update.
TRIGGER_Z_MIN = 2.25
NEIGHBOR_K = 10
MIN_NEIGHBOR_OVERLAP = 2
PAGE_RELEVANCE_MIN = 0.85
PAGE_RELEVANCE_TOP = 3
PER_TRIGGER_CAP = 2
MAX_TRIGGERS_SHOWN = 3
DECLINES_FILENAME = ".recent-declines.json"
TRIGGER_HEADINGS = frozenset({
    "what would update this page",
    "what would change the conclusion",
})
#: Proposal statuses whose decisive uncertainty is still an open question.
OPEN_PROPOSAL_STATUSES = frozenset({"proposed", "shortlisted", "deferred", "drafted"})


# ---------- seeds ----------

@dataclass
class SeedSet:
    label: str
    dois: list[str]
    available: int          # DOI-bearing seed papers before the budget cap


def _doi(page: Page) -> str | None:
    value = str(page.fm.get("doi") or "").strip().strip("\"'").lower()
    return value if value and value not in ("todo", "none") else None


def _year(page: Page) -> int:
    try:
        return int(str(page.fm.get("year") or 0)[:4])
    except ValueError:
        return 0


def _paper_pages(pages: list[Page]) -> list[Page]:
    return [p for p in pages if p.page_type == "paper" and p.category not in PAGE_TYPE_DIRS]


def _find_page(pages: list[Page], ident: str) -> Page | None:
    ident = ident.strip().removesuffix(".md")
    for p in pages:
        if p.key == ident or p.stem == ident:
            return p
    return None


_WIKILINK_TARGET_RE = re.compile(r"\[\[([^\]|#]+)")


def _cited_stems(page: Page) -> list[str]:
    """Stems a page links to, in first-seen order (inline links and footnotes)."""
    out: list[str] = []
    for m in _WIKILINK_TARGET_RE.finditer(page.body):
        stem = m.group(1).strip().rsplit("/", 1)[-1]
        if stem not in out:
            out.append(stem)
    return out


class SeedError(ValueError):
    """A seed argument named nothing usable — exit 1."""


def resolve_seeds(
    pages: list[Page],
    *,
    categories: list[str],
    page_keys: list[str],
    stems: list[str],
    budget: int,
) -> SeedSet:
    """Collect DOI-bearing paper pages named by the seed arguments.

    Over budget, the newest papers win: they sit nearest the frontier the
    recommendations are meant to extend. Ties break on stem so a re-run with
    the same corpus sends the same request (and hits the cache).
    """
    papers = _paper_pages(pages)
    chosen: dict[str, Page] = {}
    labels: list[str] = []

    valid = content_categories()
    for cat in categories:
        cat = cat.strip().lower()
        if cat not in valid:
            raise SeedError(f"unknown category {cat!r}; valid: {', '.join(sorted(valid))}")
        labels.append(cat)
        for p in papers:
            if p.category == cat:
                chosen[p.stem] = p

    by_stem = {p.stem: p for p in papers}
    for key in page_keys:
        page = _find_page(pages, key)
        if page is None:
            raise SeedError(f"no wiki page {key!r}")
        labels.append(page.key)
        for stem in _cited_stems(page):
            if stem in by_stem:
                chosen[stem] = by_stem[stem]

    for stem in stems:
        page = _find_page(papers, stem)
        if page is None:
            raise SeedError(f"no paper page {stem!r}")
        chosen[page.stem] = page
    if stems:
        labels.append(f"{len(stems)} paper(s)")

    with_doi = [p for p in chosen.values() if _doi(p)]
    ranked = sorted(with_doi, key=lambda p: (-_year(p), p.stem))[:budget]
    return SeedSet(
        label=", ".join(labels) or "(none)",
        dois=[_doi(p) for p in ranked],
        available=len(with_doi),
    )


# ---------- update triggers ----------

@dataclass
class Trigger:
    page: str               # category/stem
    page_type: str
    text: str
    cited: frozenset[str] = frozenset()   # paper stems the page links to


_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET_LINE_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+")
_FOOTNOTE_REF_RE = re.compile(r"\[\^[^\]]+\]")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_MARKUP_RE = re.compile(r"[*_`]")


_ID_SUFFIX_RE = re.compile(r"--[0-9a-f]{6,}$")


def _link_words(target: str) -> str:
    """`[[genomics/siren-2021-pangenomics-enables]]` → `siren 2021 pangenomics
    enables`: a stem is author, year and title words, which carry signal."""
    stem = target.split("#", 1)[0].rsplit("/", 1)[-1]
    return _ID_SUFFIX_RE.sub("", stem).replace("-", " ")


def _clean(text: str) -> str:
    """Prose only; a wikilink becomes its alias or its stem's words."""
    text = _FOOTNOTE_REF_RE.sub("", text)
    text = _WIKILINK_RE.sub(lambda m: m.group(2) or _link_words(m.group(1)), text)
    text = _MARKUP_RE.sub("", text)
    return " ".join(text.split())


def _units(lines: list[str], *, whole: bool) -> list[str]:
    """Split section lines into bullets (nested bullets join their parent) or
    paragraphs; `whole` returns the section as a single unit."""
    if whole:
        text = _clean(" ".join(
            _BULLET_LINE_RE.sub("", ln) for ln in lines if not _HEADING_LINE_RE.match(ln)
        ))
        return [text] if text else []
    units: list[str] = []
    buf: list[str] = []
    parent: int | None = None

    def flush() -> None:
        text = _clean(" ".join(buf))
        if text:
            units.append(text)
        buf.clear()

    for raw in lines:
        if not raw.strip() or _HEADING_LINE_RE.match(raw):
            flush()
            parent = None
            continue
        bm = _BULLET_LINE_RE.match(raw)
        if bm:
            indent = len(bm.group(1).expandtabs())
            if parent is None or indent <= parent:
                flush()
                parent = indent
            buf.append(raw[bm.end():])
            continue
        buf.append(raw)
    flush()
    return units


def _section_units(body: str, headings: frozenset[str], *, whole: bool = False) -> list[str]:
    """Units under every heading (any level) named in `headings`. A section
    runs to the next heading at the same or a higher level, so an idea page's
    `### What would change the conclusion` stops at the next `###`."""
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _HEADING_LINE_RE.match(lines[i])
        if not m or m.group(2).strip().lower() not in headings:
            i += 1
            continue
        level = len(m.group(1))
        j = i + 1
        while j < len(lines):
            m2 = _HEADING_LINE_RE.match(lines[j])
            if m2 and len(m2.group(1)) <= level:
                break
            j += 1
        out.extend(_units(lines[i + 1:j], whole=whole))
        i = j
    return out


def collect_triggers(pages: list[Page]) -> list[Trigger]:
    """Every stated condition under which a page expects to change."""
    out: list[Trigger] = []
    for p in pages:
        if p.page_type in ("synthesis", "idea"):
            texts = _section_units(p.body, TRIGGER_HEADINGS)
        elif (p.page_type == "proposal"
              and str(p.fm.get("status") or "").strip().lower() in OPEN_PROPOSAL_STATUSES):
            texts = _section_units(p.body, frozenset({"decisive uncertainty"}), whole=True)
        else:
            continue
        cited = frozenset(_cited_stems(p))
        out.extend(
            Trigger(page=p.key, page_type=p.page_type, text=t, cited=cited)
            for t in texts if len(t.split()) >= MIN_TRIGGER_WORDS
        )
    return out


# ---------- candidates ----------

@dataclass
class Candidate:
    paper_id: str | None
    doi: str | None
    title: str
    publication_date: str | None
    year: int | None
    venue: str
    citation_count: int | None
    has_abstract: bool
    fit: float | None = None
    nearest: list[dict] = field(default_factory=list)
    triggers: list[dict] = field(default_factory=list)
    first_seen: str | None = None
    #: Title + abstract, for embedding only. Deliberately absent from
    #: `to_dict`: the abstract must not be persisted or displayed.
    score_text: str = field(default="", repr=False)

    @property
    def key(self) -> str:
        return self.doi or f"s2:{self.paper_id.lower()}"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "doi": self.doi,
            "paper_id": self.paper_id,
            "title": self.title,
            "publication_date": self.publication_date,
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "has_abstract": self.has_abstract,
            "fit": self.fit,
            "nearest": self.nearest,
            "triggers": self.triggers,
            "first_seen": self.first_seen,
        }


_TITLE_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm_title(title: str) -> str:
    return _TITLE_NORM_RE.sub(" ", (title or "").lower()).strip()


def _parse_date(value: str | None) -> _dt.date | None:
    if not value:
        return None
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def filter_candidates(
    articles: list[ScholarlyArticle],
    *,
    cutoff: _dt.date,
    wiki_dois: set[str],
    wiki_titles: set[str],
    declined: set[str],
) -> tuple[list[Candidate], dict[str, int]]:
    """Drop what the user already has or has rejected, and what is too old.

    A title match counts as "in the wiki" too: S2 often lists the preprint
    of a paper whose journal version the wiki holds under a different DOI.
    An undated paper is kept when its year reaches the window and shows as
    undated, rather than being guessed into or out of it.
    """
    counts = {"returned": len(articles), "in_wiki": 0, "declined": 0,
              "outside_window": 0, "no_identifier": 0}
    out: list[Candidate] = []
    seen: set[str] = set()
    for a in articles:
        paper_id = (a.raw or {}).get("paperId") or None
        doi = a.doi_lower
        if not doi and not paper_id:
            counts["no_identifier"] += 1
            continue
        if (doi and doi in wiki_dois) or _norm_title(a.title) in wiki_titles:
            counts["in_wiki"] += 1
            continue
        cand = Candidate(
            paper_id=paper_id, doi=doi, title=a.title or "",
            publication_date=a.publication_date, year=a.year,
            venue=a.venue or "", citation_count=a.citation_count,
            has_abstract=bool(a.abstract),
            score_text="\n\n".join(x for x in (a.title, a.abstract) if x),
        )
        # S2 may add a DOI after a paper was first recommended without one.
        # Keep its S2-id decline effective when the preferred key changes.
        s2_key = f"s2:{paper_id.lower()}" if paper_id else None
        if cand.key in declined or (s2_key is not None and s2_key in declined):
            counts["declined"] += 1
            continue
        published = _parse_date(a.publication_date)
        in_window = (published >= cutoff) if published else bool(a.year and a.year >= cutoff.year)
        if not in_window:
            counts["outside_window"] += 1
            continue
        if cand.key in seen:
            continue
        seen.add(cand.key)
        out.append(cand)
    return out, counts


# ---------- local scoring ----------

def _embed(texts: list[str]) -> np.ndarray | None:
    from ..index import embeddings
    return embeddings.embed_texts(texts)


def _load_page_index() -> tuple[np.ndarray, list[dict]] | None:
    from ..index import pages_semantic
    return pages_semantic.load_index()


def score_candidates(candidates: list[Candidate], triggers: list[Trigger]) -> bool:
    """Fill `fit`, `nearest` and `triggers` in place. False when the page
    index or the embedding model is unavailable; candidates stay unscored."""
    if not candidates:
        return True
    loaded = _load_page_index()
    if loaded is None:
        return False
    arr, rows = loaded
    paper_ix = [i for i, r in enumerate(rows) if r.get("page_type") == "paper"]
    if not paper_ix:
        return False
    emb = _embed([c.score_text or c.title for c in candidates])
    if emb is None or emb.ndim != 2 or emb.shape[1] != arr.shape[1]:
        return False

    papers = arr[paper_ix]
    paper_stems = [rows[i]["stem"] for i in paper_ix]
    stem_ix = {s: n for n, s in enumerate(paper_stems)}
    sims = emb @ papers.T
    k = min(FIT_TOP_K, len(paper_ix))
    neighbours: list[set[str]] = []
    for i, cand in enumerate(candidates):
        order = np.argsort(-sims[i])
        cand.fit = round(float(sims[i][order[:k]].mean()), 4)
        cand.nearest = [
            {"key": rows[paper_ix[int(j)]]["key"], "score": round(float(sims[i][int(j)]), 4)}
            for j in order[:k]
        ]
        neighbours.append({paper_stems[int(j)] for j in order[:NEIGHBOR_K]})

    if triggers:
        temb = _embed([t.text for t in triggers])
        if temb is not None and temb.ndim == 2 and temb.shape[1] == emb.shape[1]:
            background = papers @ temb.T
            mean = background.mean(axis=0)
            sd = np.maximum(background.std(axis=0), 1e-6)
            tsims = emb @ temb.T
            zs = (tsims - mean) / sd
            relevance: dict[tuple[int, str], float] = {}

            def page_relevance(i: int, trig: Trigger) -> float:
                key = (i, trig.page)
                if key not in relevance:
                    cols = [stem_ix[s] for s in trig.cited if s in stem_ix]
                    if len(cols) < PAGE_RELEVANCE_TOP:
                        relevance[key] = float("-inf")
                    else:
                        near = np.sort(sims[i, cols])[-PAGE_RELEVANCE_TOP:]
                        relevance[key] = float(near.mean())
                return relevance[key]

            # The floor applies before the per-trigger cap, so a slot is not
            # spent on a paper that the floor would then discard.
            passing: dict[int, list[tuple[float, int]]] = {}
            for i, j in zip(*np.nonzero(zs >= TRIGGER_Z_MIN)):
                i, j = int(i), int(j)
                trig = triggers[j]
                if len(neighbours[i] & trig.cited) < MIN_NEIGHBOR_OVERLAP:
                    continue
                if page_relevance(i, trig) < PAGE_RELEVANCE_MIN:
                    continue
                passing.setdefault(j, []).append((float(zs[i, j]), i))
            best: dict[int, dict[str, tuple[float, float, int, Trigger]]] = {}
            for j, hits in passing.items():
                trig = triggers[j]
                for z, i in sorted(hits, reverse=True)[:PER_TRIGGER_CAP]:
                    mine = best.setdefault(i, {})
                    if trig.page not in mine or z > mine[trig.page][0]:
                        overlap = len(neighbours[i] & trig.cited)
                        mine[trig.page] = (z, float(tsims[i, j]), overlap, trig)
            for i, mine in best.items():
                candidates[i].triggers = [
                    {"page": t.page, "page_type": t.page_type, "z": round(z, 2),
                     "score": round(s, 4), "neighbor_overlap": ov,
                     "page_relevance": round(relevance[(i, t.page)], 4), "text": t.text}
                    for z, s, ov, t in sorted(mine.values(), key=lambda x: -x[0])
                ][:MAX_TRIGGERS_SHOWN]
    return True


def rank(candidates: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """(possible page updates, everything else), each best-first."""
    def fit(c: Candidate) -> float:
        return c.fit if c.fit is not None else -1.0
    updates = sorted((c for c in candidates if c.triggers),
                     key=lambda c: (-c.triggers[0]["z"], -fit(c)))
    rest = sorted((c for c in candidates if not c.triggers), key=lambda c: -fit(c))
    return updates, rest


# ---------- declines and first-seen ledger ----------

def _declines_path() -> Path:
    return wiki_root() / DECLINES_FILENAME


#: A Semantic Scholar paper page. The report prints `/paper/<id>`; the site's
#: own links, and so a URL copied from the address bar, put a title slug first:
#: `/paper/<Title-Words-Author>/<id>`. The id is always the last segment.
_S2_PAPER_URL_RE = re.compile(
    r"https?://(?:www\.)?semanticscholar\.org/paper/(?:[^/?#]+/)?([^/?#]+)/?(?:[?#].*)?",
    flags=re.IGNORECASE,
)
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")


class DeclineKeyError(ValueError):
    """A decline argument that names no paper — exit 1."""


def normalize_key(value: str) -> str:
    """One key per paper: a lowercase DOI, or `s2:<id>` for a paper without one.

    Accepts DOI spellings (`10.1/X`, `doi:10.1/X`, `https://doi.org/10.1/X`),
    Semantic Scholar paper URLs with or without the title slug, and `s2:<id>`.
    S2 ids are lowercase hex, which is how candidates are keyed, so the id is
    lowercased too. Raises `DeclineKeyError` for anything else: stored as-is,
    such a key would match no candidate and be sent to S2 as a bogus negative
    DOI, a decline that reports success and does nothing.
    """
    v = value.strip()
    low = v.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if low.startswith(prefix):
            low = low[len(prefix):]
            break
    else:
        s2_url = _S2_PAPER_URL_RE.fullmatch(v)
        if s2_url:
            return "s2:" + s2_url.group(1).lower()
        if low.startswith("s2:") and len(low) > 3:
            return low
    if _DOI_RE.fullmatch(low):
        return low
    raise DeclineKeyError(
        f"{value!r} is not a DOI, a Semantic Scholar paper URL, or s2:<paper-id>"
    )


def load_declines() -> dict[str, dict]:
    """Read declines under canonical keys, including URLs saved by old CLI builds.

    Keep unrecognized legacy keys visible in `--list-declined`; they cannot
    match a candidate, and `negative_dois` must not send them to S2.
    """
    data = read_json(_declines_path(), {})
    if not isinstance(data, dict):
        return {}
    declines: dict[str, dict] = {}
    for raw_key, entry in data.items():
        if not isinstance(raw_key, str) or not isinstance(entry, dict):
            continue
        try:
            key = normalize_key(raw_key)
        except DeclineKeyError:
            key = raw_key
        prior = declines.get(key)
        if prior is None or str(entry.get("declined_at") or "") >= str(prior.get("declined_at") or ""):
            declines[key] = entry
    return declines


def add_decline(key: str, reason: str) -> str:
    """Permanent until removed: a paper does not become relevant with time."""
    key = normalize_key(key)
    declines = load_declines()
    declines[key] = {"reason": reason, "declined_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json_atomic(_declines_path(), declines)
    return key


def _stored_key(value: str, declines: dict[str, dict]) -> str:
    """The key `value` names in `declines`.

    Normally the canonical key. A legacy entry that `normalize_key` cannot
    read keeps its raw key in `load_declines`, so it is also matched verbatim
    as `--list-declined` prints it; otherwise it could be listed but never
    removed. Raises `DeclineKeyError` when `value` is neither.
    """
    try:
        return normalize_key(value)
    except DeclineKeyError:
        raw = value.strip()
        for candidate in (raw, raw.lower()):
            if candidate in declines:
                return candidate
        raise


def remove_decline(key: str) -> bool:
    declines = load_declines()
    key = _stored_key(key, declines)
    if key not in declines:
        return False
    del declines[key]
    write_json_atomic(_declines_path(), declines)
    return True


def negative_dois(declines: dict[str, dict], cap: int = MAX_NEGATIVES) -> list[str]:
    """Most recently declined DOIs first; S2 and malformed legacy keys stay local."""
    dois = [(v.get("declined_at", ""), k) for k, v in declines.items()
            if _DOI_RE.fullmatch(k)]
    return [k for _, k in sorted(dois, reverse=True)[:cap]]


def _recent_dir() -> Path:
    return s2_cache_dir() / "recent"


def mark_first_seen(candidates: list[Candidate], today: _dt.date) -> None:
    """Stamp each candidate with the day it first surfaced, so a re-run shows
    what is new since the last look. A cache file: deleting it just makes
    everything new again."""
    path = _recent_dir() / "seen.json"
    seen = read_json(path, {})
    if not isinstance(seen, dict):
        seen = {}
    for cand in candidates:
        cand.first_seen = seen.setdefault(cand.key, today.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, seen)


# ---------- orchestration ----------

def run(
    *,
    categories: list[str] = (),
    page_keys: list[str] = (),
    stems: list[str] = (),
    since: _dt.date | None = None,
    days: int = DEFAULT_DAYS,
    max_age_days: int = 1,
    today: _dt.date | None = None,
    provider: SemanticScholarProvider | None = None,
) -> dict:
    today = today or _dt.date.today()
    cutoff = since or (today - _dt.timedelta(days=days))
    pages = read_pages()
    declines = load_declines()
    negatives = negative_dois(declines)
    seeds = resolve_seeds(
        pages, categories=list(categories), page_keys=list(page_keys),
        stems=list(stems), budget=MAX_RECOMMENDATION_SEEDS - len(negatives),
    )
    if not seeds.dois:
        raise SeedError(f"no DOI-bearing paper pages among the seeds ({seeds.label})")

    provider = provider or SemanticScholarProvider(
        log_tag=LOG_TAG, force_refresh_days=max_age_days,
    )
    articles = provider.get_recommendations_for_seeds(seeds.dois, negatives, limit=POOL)

    papers = _paper_pages(pages)
    wiki_dois = {d for p in papers if (d := _doi(p))}
    wiki_titles = {_norm_title(str(p.fm.get("title") or "")) for p in papers} - {""}
    candidates, counts = filter_candidates(
        articles, cutoff=cutoff, wiki_dois=wiki_dois,
        wiki_titles=wiki_titles, declined=set(declines),
    )
    triggers = collect_triggers(pages)
    scored = score_candidates(candidates, triggers)
    if not scored:
        log("WARN: page index or embedding model unavailable — candidates are "
            "unranked and unmatched; run `researchwiki reindex`", tag=LOG_TAG)
    mark_first_seen(candidates, today)
    updates, rest = rank(candidates)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
        "provider": "semantic-scholar",
        "seeds": {"label": seeds.label, "used": len(seeds.dois),
                  "available": seeds.available, "negatives": len(negatives)},
        "since": cutoff.isoformat(),
        "counts": {**counts, "candidates": len(candidates)},
        "scored": scored,
        "trigger_gates": {"z_min": TRIGGER_Z_MIN, "neighbor_k": NEIGHBOR_K,
                          "min_neighbor_overlap": MIN_NEIGHBOR_OVERLAP,
                          "page_relevance_min": PAGE_RELEVANCE_MIN,
                          "per_trigger_cap": PER_TRIGGER_CAP},
        "triggers_indexed": len(triggers),
        "page_matches": [c.to_dict() for c in updates],
        "candidates": [c.to_dict() for c in rest],
    }
    digest = hashlib.md5("|".join(seeds.dois).encode()).hexdigest()[:8]
    out = _recent_dir() / f"{today.isoformat()}__{digest}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, snapshot)
    snapshot["snapshot_path"] = str(out)
    return snapshot


# ---------- rendering ----------

def _row(c: dict, today: str) -> list[str]:
    new = "NEW " if c.get("first_seen") == today else "    "
    date = c.get("publication_date") or f"{c.get('year') or '?'} (undated)"
    ident = c.get("doi") or f"https://www.semanticscholar.org/paper/{c['paper_id']}"
    fit = f"fit {c['fit']:.2f}" if c.get("fit") is not None else "fit  —  "
    venue = c.get("venue") or "—"
    lines = [f"- {new}{date}  {fit}  `{ident}`  {venue}", f"    {c.get('title') or '(no title)'}"]
    if c.get("nearest"):
        top = c["nearest"][0]
        lines.append(f"    nearest: [[{top['key']}]] ({top['score']:.2f})")
    for t in c.get("triggers") or []:
        text = t["text"] if len(t["text"]) <= 160 else t["text"][:157] + "…"
        lines.append(f"    ↳ on-topic for [[{t['page']}]]; closest trigger (z {t['z']:.1f}): {text}")
    return lines


def render(snapshot: dict, limit: int) -> str:
    today = snapshot["generated_at"][:10]
    s, n = snapshot["seeds"], snapshot["counts"]
    out = [f"# Recent papers near {s['label']}", ""]
    out.append(
        f"_Seeded with {s['used']} of {s['available']} DOI-bearing paper(s)"
        + (f" and {s['negatives']} declined" if s["negatives"] else "")
        + f"; published since {snapshot['since']}. S2 returned {n['returned']}: "
        f"{n['in_wiki']} already in the wiki, {n['declined']} declined, "
        f"{n['outside_window']} outside the window._"
    )
    out.append("")
    all_updates = snapshot["page_matches"]
    updates = all_updates[:max(0, limit)]
    rest = snapshot["candidates"][:max(0, limit - len(updates))]
    out.append(f"## Near a page's open questions ({len(updates)} of {len(all_updates)})")
    out.extend(line for c in updates for line in _row(c, today))
    if not all_updates:
        out.append("_(none above the trigger threshold)_")
    out.append("")
    out.append(f"## Closest to the corpus ({len(rest)} of {len(snapshot['candidates'])})")
    out.extend(line for c in rest for line in _row(c, today))
    out.append("")
    out.append("_Leads only: nothing here is evidence until its PDF is in `inbox/` and "
               "ingested. Reject one for good with "
               "`researchwiki scout recent --decline <DOI-or-S2-URL> --reason \"…\"`._")
    return "\n".join(out)


# ---------- CLI ----------

def main(argv: list[str], *, prog: str = "researchwiki scout recent") -> int:
    if argv and argv[0] == "report":
        from . import recent_report
        return recent_report.main(argv[1:])
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Recently published papers near a category, page, or paper set, "
                    "ranked locally against the wiki. Structured S2 metadata only; "
                    "no model calls.",
    )
    seeds = ap.add_argument_group("seeds (at least one)")
    seeds.add_argument("--category", action="append", default=[],
                       help="Seed with this category's paper pages (repeatable).")
    seeds.add_argument("--page", action="append", default=[],
                       help="Seed with the papers a synthesis/idea page cites (repeatable).")
    seeds.add_argument("--papers", nargs="+", default=[], metavar="STEM",
                       help="Seed with these paper stems.")
    window = ap.add_mutually_exclusive_group()
    window.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Publication window in days (default {DEFAULT_DAYS}).")
    window.add_argument("--since", help="Publication window start, YYYY-MM-DD.")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"Rows shown, page matches first (default {DEFAULT_LIMIT}).")
    ap.add_argument("--max-age-days", type=int, default=1,
                    help="Re-poll S2 when this seed set's cached answer is older "
                         "than N days; 0 always re-polls (default 1).")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="Emit the snapshot as JSON.")
    manage = ap.add_argument_group("declines")
    manage.add_argument("--decline", metavar="DOI_OR_S2_URL",
                        help="Never show this paper again; DOI declines also steer S2 away from it.")
    manage.add_argument("--reason", help="Why (required with --decline).")
    manage.add_argument("--undecline", metavar="DOI_OR_S2_URL",
                        help="Remove a decline; same forms as --decline, or a key "
                             "exactly as --list-declined prints it.")
    manage.add_argument("--list-declined", action="store_true")
    args = ap.parse_args(argv)

    if args.decline or args.undecline:
        try:
            key = (normalize_key(args.decline) if args.decline
                   else _stored_key(args.undecline, load_declines()))
        except DeclineKeyError as exc:
            print(f"{prog}: {exc}", file=sys.stderr)
            return 1
    if args.decline:
        if not (args.reason or "").strip():
            print(f"{prog}: --decline needs --reason", file=sys.stderr)
            return 1
        print(f"declined {add_decline(key, args.reason.strip())}")
        return 0
    if args.undecline:
        if remove_decline(key):
            print(f"undeclined {key}")
            return 0
        print(f"{prog}: {key} was not declined", file=sys.stderr)
        return 1
    if args.list_declined:
        declines = load_declines()
        if args.as_json:
            print(json.dumps(declines, indent=2))
        else:
            for key, entry in sorted(declines.items()):
                print(f"{key}  ({entry.get('declined_at', '?')}): {entry.get('reason', '')}")
        return 0

    if not (args.category or args.page or args.papers):
        print(f"{prog}: choose seeds with --category, --page, or --papers", file=sys.stderr)
        return 1
    since = None
    if args.since:
        since = _parse_date(args.since)
        if since is None:
            print(f"{prog}: invalid --since {args.since!r}; expected YYYY-MM-DD", file=sys.stderr)
            return 1
    if args.max_age_days < 0 or args.days <= 0:
        print(f"{prog}: --days must be positive and --max-age-days non-negative",
              file=sys.stderr)
        return 1

    try:
        snapshot = run(
            categories=args.category, page_keys=args.page, stems=args.papers,
            since=since, days=args.days, max_age_days=args.max_age_days,
        )
    except SeedError as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    else:
        print(render(snapshot, args.limit))
    return 0
