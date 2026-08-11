"""Reconcile phase — resolve canonical metadata from PDF + S2/Crossref.

This is the first phase in the ingest state machine. Outputs a metadata dict
that downstream phases (extract, author, grade, commit) consume. All PDF /
metadata / author / title / year extraction helpers live here too — the
external surface is the single `reconcile` function.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ... import metadata_sanity
from ...pdf.text import detect_doi, extract_pdf, find_url_doi_candidates, pdf_shape
from ...providers.crossref import crossref_structural_signals, verify_doi_via_crossref
from ...providers.semantic_scholar import SemanticScholarProvider
from ...stems import derive_stem
from .. import llm
from ...log import log


# JSON Schema for the reconcile/extractor response. Honored by the chat-relay
# provider (validates + retries on mismatch); other providers ignore it.
# Required keys mirror _empty_metadata; each is nullable since the extractor
# is told to emit null when not confident.
_RECONCILE_SCHEMA = {
    "type": "object",
    "required": ["title", "first_author_surname", "all_authors", "year",
                 "doi", "venue_hint", "paper_type", "abstract"],
    "properties": {
        "title":               {"type": ["string", "null"]},
        "first_author_surname": {"type": ["string", "null"]},
        "all_authors":         {"type": "array", "items": {"type": "string"}},
        "year":                {"type": ["integer", "null"]},
        "doi":                 {"type": ["string", "null"]},
        "venue_hint":          {"type": ["string", "null"]},
        "paper_type":          {"type": ["string", "null"]},
        "abstract":            {"type": "string"},
    },
}


_LLM_RECONCILE_SYSTEM = """\
You extract paper metadata from the first 1-2 pages of a research-paper
PDF. Return strictly-formatted JSON. Be conservative: emit `null` for
any field you cannot confidently extract from the visible text.

Fields to extract:
- "title": full paper title as printed on the first page (no
  truncation, no abbreviation, no subtitles dropped). Strip
  trailing/leading whitespace. Null if you can't tell.
- "first_author_surname": surname of the first listed author, with
  diacritics preserved. No affiliation marks (digits / asterisks /
  daggers / superscripts). Null if you can't tell.
- "all_authors": list of full author names in order, with affiliation
  marks stripped. Empty list if you can't tell.
- "year": 4-digit publication year as printed on the paper itself —
  the paper's own year, NOT a year from a citation in the references.
  For preprints, the version-posted year (e.g., bioRxiv "this version
  posted February 14, 2026" → 2026). Integer or null.
- "doi": the paper's own DOI, lowercase. NOT a DOI cited from another
  paper. If the visible text shows multiple DOIs (e.g., a citation
  bundle), pick only the one belonging to THIS paper. Null if you
  cannot tell with high confidence — DO NOT guess.
- "venue_hint": journal/conference name as printed (e.g., "Nature",
  "bioRxiv", "Bioinformatics"). Null if not visible.
- "paper_type": one of "research", "review", "perspective",
  "methods", "preprint", "clinical-trial", or null if you can't tell.
  Use "clinical-trial" when an NCT ID, EudraCT ID, "ClinicalTrials.gov"
  reference, or explicit phase 1/2/3 trial language is present.
- "abstract": verbatim abstract text from the first page (no
  paraphrasing, no summarization). Cap at 1500 characters — truncate
  with "..." if the source is longer. Empty string if not visible.

Important:
- DO NOT fabricate DOIs from priors. If the paper is too recent or
  not in your training data, return null.
- DO NOT include affiliation markers (∗, †, ‡, §, ¶, digits) in
  surname or author names.
- DO NOT shorten the title (no "...").
- Respond with JSON only — no prose, no markdown code fences."""


def propose_metadata_llm(pdf_text: str, *, use_stub: bool = False) -> dict:
    """Extract metadata from first-2-pages PDF text via the configured
    `extractor` model (see `config/models.yaml`).

    Returns a dict with each field either populated or `None`. The LLM
    is instructed to emit `None` when not confident — the failure mode
    we guard against is fabricated DOIs on fresh preprints not in the
    training data.

    Caller is responsible for downstream validation: cross-check DOI
    against S2 to confirm it resolves; cross-check year against the
    PDF's bioRxiv watermark when present.
    """
    head = pdf_text[:6000]
    if not head.strip():
        return _empty_metadata()

    if use_stub:
        return _empty_metadata()

    prompt = f"# Paper text (first 1-2 pages)\n\n{head}"
    try:
        resp = llm.call(
            phase="reconcile",
            prompt=prompt,
            system=_LLM_RECONCILE_SYSTEM,
            use_stub=False,
            schema=_RECONCILE_SCHEMA,
        )
    except Exception as e:
        log(f"call failed: {e}", tag="reconcile-llm")
        return _empty_metadata()

    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return _empty_metadata()
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return _empty_metadata()
    if not isinstance(out, dict):
        return _empty_metadata()

    return _coerce_metadata_shape(out)


def _empty_metadata() -> dict:
    return {
        "title": None,
        "first_author_surname": None,
        "all_authors": [],
        "year": None,
        "doi": None,
        "venue_hint": None,
        "paper_type": None,
        "abstract": "",
    }


def _coerce_metadata_shape(d: dict) -> dict:
    """Normalize LLM output: clip strings, coerce year to int, drop empty
    surrogates, lowercase DOI. Tolerates the LLM emitting either string
    "null" or actual nulls."""
    def _norm_str(v) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s and s.lower() not in ("null", "none", "n/a", "") else None

    out = _empty_metadata()
    out["title"] = _norm_str(d.get("title"))
    out["first_author_surname"] = _norm_str(d.get("first_author_surname"))
    raw_authors = d.get("all_authors") or []
    if isinstance(raw_authors, list):
        out["all_authors"] = [s for s in (_norm_str(a) for a in raw_authors) if s]
    year = d.get("year")
    if isinstance(year, int) and 1900 <= year <= 2099:
        out["year"] = year
    elif isinstance(year, str):
        try:
            y = int(year.strip())
            if 1900 <= y <= 2099:
                out["year"] = y
        except (ValueError, TypeError):
            pass
    doi = _norm_str(d.get("doi"))
    out["doi"] = doi.lower() if doi else None
    out["venue_hint"] = _norm_str(d.get("venue_hint"))
    pt = _norm_str(d.get("paper_type"))
    if pt and pt.lower() in ("research", "review", "perspective", "methods", "preprint"):
        out["paper_type"] = pt.lower()
    abs_text = d.get("abstract") or ""
    out["abstract"] = str(abs_text).strip()
    return out


# Preprint-server names that should NOT be recorded as a paper's venue when a
# journal of record exists. A deposited preprint/accepted-manuscript PDF prints
# one of these on its masthead, so the LLM's `venue_hint` (and sometimes S2's
# `venue` field) reads the preprint server rather than the journal.
_PREPRINT_VENUES = (
    "biorxiv", "medrxiv", "arxiv", "chemrxiv", "research square",
    "researchsquare", "ssrn", "preprints.org", "preprint", "zenodo",
)


# Venue plausibility lives in `metadata_sanity` so `lint`'s `venue_suspect`
# check applies the identical rule without importing the agents package (the
# import-cost constraint `tasks.lint` documents elsewhere). See that module for
# why the list is restricted to unambiguous furniture.
_is_venue_furniture = metadata_sanity.is_venue_furniture


def _s2_record_is_preprint(s2_meta: dict, doi: str | None) -> bool:
    """True when S2's record describes a preprint but our DOI is a journal DOI.

    The mismatch means S2 resolved the journal DOI to the earlier deposit's
    metadata, so its year (and title, and author list) belong to a different
    version of the paper than the PDF in hand.

    Both halves are required. An actual preprint ingest — bioRxiv PDF, bioRxiv
    DOI — must keep S2's year, which is correct and is exactly what the wiki
    wants for a preprint page.
    """
    if not _is_preprint_venue(s2_meta.get("venue")):
        return False
    d = (doi or "").lower()
    if not d:
        return False
    return not d.startswith(("10.1101/", "10.64898/", "10.48550/",
                             "10.31219/", "10.20944/", "10.2139/"))


def _is_preprint_venue(venue: str | None) -> bool:
    """True when `venue` names a preprint server rather than a journal."""
    v = (venue or "").lower()
    return bool(v) and any(p in v for p in _PREPRINT_VENUES)


def reconcile_metadata(
    pdf_path: Path,
    *,
    doi_override: str | None = None,
    title_override: str | None = None,
    year_override: int | None = None,
    authors_override: list[str] | None = None,
    use_llm: bool = False,
) -> dict:
    """Resolve canonical metadata from PDF + S2/Crossref.

    Returns a dict with keys: stem, title, year, doi, venue, authors,
    pdf_text (first 4000 chars), sources (list of which providers were
    queried). Failures fall through gracefully to PDF-only metadata.

    Title-DOI consistency check: when the in-text DOI scan picks up a DOI
    from a neighboring article (common in NEJM Perspectives, Nature N&V,
    Science First Releases — short pieces typeset alongside others), S2's
    DOI-resolved title won't match the publisher's PDF /Title metadata.
    When that happens, we fall back to title-search and override the DOI.

    `doi_override` and `title_override` are escape hatches for when the
    automatic recovery still picks the wrong paper — see `agent ingest
    --doi` / `--title`.
    """
    text, pdf_meta = extract_pdf(pdf_path, max_pages=80)

    sources: list[str] = ["pdf"]
    pdf_meta_title = _clean_pdf_meta_value(pdf_meta.get("/Title"), kind="title")
    pdf_first_page_title = _extract_title_from_pdf(pdf_meta, text)

    # LLM-reconcile (R3 default-on; --no-llm-reconcile to opt out). Run
    # a Haiku-class model over first-2-pages PDF text and use its output
    # as primary source. Regex helpers stay as fallback for any field the
    # LLM marked null. S2 lookup still wins for {title, year, authors,
    # venue} when DOI resolves — LLM-reconcile changes the EXTRACTION
    # path, not the trust ordering. ~10-paper dogfood across the 2026-05
    # batch showed 0 fabrications and 1 DOI recovery vs the regex path;
    # see plans/strengthen-reconcile.md for the full rollout.
    llm_meta: dict = _empty_metadata()
    if use_llm:
        llm_meta = propose_metadata_llm(text)
        if any(llm_meta[k] for k in ("doi", "title", "first_author_surname", "year")):
            sources.append("llm-extractor")

    seed_title = title_override or llm_meta["title"] or pdf_meta_title or pdf_first_page_title

    detected_doi = doi_override or llm_meta["doi"] or detect_doi(pdf_meta, text)
    # A template placeholder is DOI-shaped and resolves nowhere. Faithful
    # extraction is not enough: the ACM sample class prints
    # `10.1145/nnnnnnn.nnnnnnn`, and one page shipped with it.
    if detected_doi and metadata_sanity.is_placeholder_doi(detected_doi):
        log(f"doi ✗ {detected_doi} is a template placeholder, not an identifier",
            tag="reconcile")
        detected_doi = None

    # URL-DOI hunt waterfall: when the canonical-form DOI scanners come up
    # empty (no override, LLM didn't find one, no `10.X/Y` in pypdf-meta or
    # first-page text, no arXiv ID in the first 4000 chars), try preprint-
    # server URL forms whose mapping to a DOI is mechanical:
    #   ssrn.com/abstract=NNNN   → 10.2139/ssrn.NNNN
    #   arxiv.org/abs/YYMM.NNNNN → 10.48550/arXiv.YYMM.NNNNN
    # Each candidate is validated against Crossref before adoption — the URL
    # itself doesn't prove the DOI resolves; a typo'd SSRN URL would map to
    # a non-existent DOI. First validated candidate wins. Crossref indexes
    # SSRN/arXiv DOIs that Semantic Scholar misses for fresh preprints —
    # this fills the gap that produced the Jang 2025 (sub-1% somatic WGS)
    # silent reconcile failure.
    #
    # The Crossref response is also stashed in `cr_meta_from_hunt` and used
    # below as a title/year fallback when the subsequent S2 lookup returns
    # nothing — typical for fresh SSRN preprints not yet in S2's index, and
    # the only way to recover a year for PDFs whose first page lacks an
    # `Accepted:`/`Published:`/bioRxiv-watermark anchor.
    cr_meta_from_hunt: dict = {}
    if not detected_doi:
        candidates = find_url_doi_candidates(text)
        if candidates:
            log(
                f"doi-hunt → {len(candidates)} URL candidate(s): "
                f"{[f'{prov}={d}' for prov, d in candidates[:3]]}", tag="reconcile"
            )
            for provenance, candidate_doi in candidates:
                cr_meta = verify_doi_via_crossref(candidate_doi)
                if not cr_meta:
                    log(f"doi-hunt ✗ {provenance} → {candidate_doi} (crossref miss)", tag="reconcile")
                    continue
                # Resolving is not the same question as belonging. A DOI
                # scavenged out of the PDF may be a *cited* paper's, in which
                # case Crossref happily confirms it exists — see
                # `metadata_sanity`'s docstring for the case that motivated
                # this. Reject unless the resolved record describes this paper.
                why = metadata_sanity.reject_reason(
                    cr_meta.get("year"),
                    list(cr_meta.get("authors") or []),
                    cr_meta.get("title") or "",
                    llm_meta.get("first_author_surname") or "",
                    llm_meta.get("year"),
                    seed_title or "",
                )
                if why:
                    log(f"doi-hunt ✗ {provenance} → {candidate_doi} "
                        f"resolves but is a different paper: {why}", tag="reconcile")
                    continue
                detected_doi = cr_meta["doi"]
                cr_meta_from_hunt = cr_meta
                sources.append(f"crossref-doi-hunt:{provenance}")
                log(
                    f"doi-hunt ✓ {provenance} → {detected_doi} "
                    f"(crossref: {(cr_meta.get('title') or '')[:60]!r}, "
                    f"year={cr_meta.get('year')})", tag="reconcile"
                )
                break

    # Title-search fallback. The URL hunt can only find a DOI the PDF happens to
    # print; when the masthead carries none (or the only ones present belong to
    # cited work and were just rejected above), a Semantic Scholar title match
    # still finds it — this is exactly how `backfill doi` recovered
    # `10.1126/sciadv.aax9249` for the DeepSpCas9 paper after the hunt came up
    # empty. Same adoption gate as everything else here, so a near-miss title
    # can't smuggle in the wrong record. Whitelisted fields only (Rule 1).
    if not detected_doi and seed_title:
        s2_doi = _doi_via_title_search(
            seed_title,
            llm_meta.get("first_author_surname") or "",
            llm_meta.get("year"),
        )
        if s2_doi:
            detected_doi = s2_doi
            sources.append("s2-title-match")
            log(f"doi-hunt ✓ s2-title-match → {s2_doi}", tag="reconcile")

    doi = detected_doi

    # Prior-attempt detection: log a heads-up when the DOI is already in the
    # wiki. promote_to_wiki has the authoritative collision handler; this
    # just surfaces the duplicate at reconcile time instead of at commit.
    prior_stem: str | None = None
    if doi:
        try:
            from ...db import find_by_doi
            prior = find_by_doi(doi)
        except Exception:
            prior = None
        if prior:
            from datetime import datetime as _dt
            indexed = _dt.fromtimestamp(prior["indexed_at"]).strftime("%Y-%m-%d")
            prior_stem = prior["stem"]
            log(
                f"prior page → {prior['category']}/{prior['stem']} "
                f"(indexed {indexed}). promote will classify the collision.", tag="reconcile"
            )

    s2_meta: dict = {}
    provider: SemanticScholarProvider | None = None
    if doi:
        try:
            provider = SemanticScholarProvider()
            article = provider.get_by_doi(doi)
            if article is not None:
                s2_meta = {
                    "title": article.title or None,
                    "year": article.year,
                    "authors": article.authors,
                    "venue": article.venue or None,
                }
                sources.append("s2")
        except Exception as e:
            # S2 enrichment is best-effort; ingest proceeds on PDF/Crossref
            # metadata. Log so a persistent lookup failure is diagnosable
            # rather than silently degrading every ingest.
            log(f"reconcile ⚠ S2 lookup by DOI failed: "
                  f"{type(e).__name__}: {e}", tag="agent")

    # Title-DOI consistency check: only kicks in when the user didn't
    # override (otherwise trust the user) AND we have both an S2 title
    # and a publisher-provided pdf_meta_title to compare.
    if (
        not doi_override
        and s2_meta.get("title")
        and pdf_meta_title
        and _title_similarity(s2_meta["title"], pdf_meta_title) < 0.6
    ):
        log(
            f"reconcile ⚠ DOI/title mismatch:\n"
            f"             DOI={doi!r}\n"
            f"             S2 title for that DOI: {s2_meta['title'][:60]!r}\n"
            f"             pypdf /Title:          {pdf_meta_title[:60]!r}\n"
            f"             → searching S2 by pypdf title instead", tag="agent"
        )
        try:
            if provider is None:
                provider = SemanticScholarProvider()
            recovered = provider.search_by_title(pdf_meta_title)
            if recovered and _title_similarity(recovered.title or "", pdf_meta_title) >= 0.6:
                s2_meta = {
                    "title": recovered.title,
                    "year": recovered.year,
                    "authors": recovered.authors,
                    "venue": recovered.venue or None,
                }
                doi = recovered.doi or doi
                sources.append("s2-title-recovery")
                log(f"reconcile ✓ recovered DOI={doi!r}", tag="agent")
        except Exception as e:
            log(f"reconcile ⚠ S2 title-recovery search failed: "
                  f"{type(e).__name__}: {e}", tag="agent")

    title = s2_meta.get("title") or cr_meta_from_hunt.get("title") or seed_title
    # Year resolution. S2 is first — except when S2's own record is describing
    # the *preprint* rather than the journal article we ingested. That happens
    # routinely: for `10.1038/s41592-025-02626-1` (LigandMPNN, Nature Methods,
    # April 2025) S2 returns year=2023 venue=bioRxiv, the 2023 deposit's
    # metadata under the 2025 DOI. Taken at face value it produced the stem
    # `dauparas-2023-…` for a 2025 paper — and the stem is the filename and every
    # wikilink, so this is not a cosmetic field.
    #
    # The tell is already computed one block down for venue: S2 naming a preprint
    # server while the resolved DOI is *not* a preprint DOI means S2 is describing
    # the preprint. `_is_preprint_venue` rescued the venue here (the page landed
    # `Nature Methods` via the Crossref fallback) while nothing rescued the year.
    # Same signal, same conclusion — distrust the year too, and fall through to
    # Crossref and then the PDF, which CLAUDE.md already names as the source of
    # truth for the naming fields.
    s2_year = s2_meta.get("year")
    if s2_year and _s2_record_is_preprint(s2_meta, doi):
        log(f"year ✗ S2 says {s2_year} with venue "
            f"{s2_meta.get('venue')!r} for a non-preprint DOI — its record is the "
            f"preprint's; deferring to Crossref/PDF", tag="reconcile")
        s2_year = None
    # The guard above needs S2 to *admit* it is describing a preprint by naming a
    # preprint venue. S2 also merges the two versions the other way round: it keeps
    # the journal's venue and the preprint's year, and then nothing above fires.
    # Observed 2026-08-10 on minimap2 — for the journal DOI
    # `10.1093/bioinformatics/bty191` S2 returns year=2017 venue='Bioinform.' with
    # `ArXiv: 1708.01492` (posted 2017-08) among its externalIds, while the PDF
    # prints "accepted on May 4, 2018" throughout. The reconcile prompt asks the
    # LLM for "the paper's own year ... NOT a year from a citation", it answered
    # 2018 correctly, and the chain below discarded it because S2 sits two places
    # higher. Result: stem `li-2017-…` for a 2018 paper.
    #
    # So when S2 and the document disagree, ask Crossref to arbitrate rather than
    # trusting either — Crossref's record for a journal DOI is the journal's own,
    # with no preprint to merge. Only fires on genuine disagreement, so the common
    # path adds no request, and responses are cached under `.crossref-cache/`.
    # Fail-safe: no Crossref answer leaves S2's year standing, exactly as before.
    if s2_year and doi and llm_meta["year"] and llm_meta["year"] != s2_year:
        cr_year = (verify_doi_via_crossref(doi) or {}).get("year")
        if cr_year and cr_year != s2_year:
            log(f"year ✗ S2 says {s2_year} but the PDF says {llm_meta['year']} — "
                f"Crossref says {cr_year} for {doi}; S2's record carries the "
                f"preprint's year under the journal's venue, taking Crossref",
                tag="reconcile")
            s2_year = cr_year
    year = (
        year_override
        or s2_year
        or cr_meta_from_hunt.get("year")
        or llm_meta["year"]
        or _extract_year_from_pdf(text, doi=doi)
    )
    # Venue resolution. S2 is first, but for preprint/accepted-manuscript PDFs
    # S2 often has no venue (404s for fresh/Cell DOIs) — and the next fallback,
    # the LLM's `venue_hint`, reads the PDF masthead, which on such PDFs prints
    # "bioRxiv" or nothing. So when we have a DOI and the venue so far is missing
    # OR names a preprint server, consult Crossref's `container-title` (the
    # journal of record) before trusting the masthead. cr_meta_from_hunt is only
    # populated on the URL→DOI hunt path; for a directly-detected DOI we fetch
    # Crossref here (responses are cached under .crossref-cache/).
    venue = s2_meta.get("venue") or cr_meta_from_hunt.get("venue")
    if doi and (not venue or _is_preprint_venue(venue)):
        cr_venue = cr_meta_from_hunt.get("venue") or (verify_doi_via_crossref(doi) or {}).get("venue")
        if cr_venue and not _is_preprint_venue(cr_venue):
            venue = cr_venue
    if not venue and llm_meta["venue_hint"]:
        hint = llm_meta["venue_hint"]
        if _is_venue_furniture(hint):
            log(f"venue ✗ masthead hint {hint!r} is page furniture, not a journal "
                f"— leaving venue unset", tag="reconcile")
        else:
            venue = hint
    # Authors: trust S2 when we have it (clean list). When S2 returns
    # nothing (common for fresh arXiv preprints not yet indexed), prefer
    # pypdf's /Author metadata field over body-text scanning — the
    # publisher's typesetting tool puts authors there reliably, while
    # the text scanner can grab sentence fragments from the abstract or
    # captions when the author block is interleaved with affiliations.
    pdf_meta_author_str = _clean_pdf_meta_value(pdf_meta.get("/Author"), kind="author")
    if authors_override:
        raw_authors = authors_override
        authors_origin = "override"
    elif s2_meta.get("authors"):
        raw_authors = s2_meta["authors"]
        authors_origin = "s2"
    elif llm_meta["all_authors"]:
        raw_authors = llm_meta["all_authors"]
        authors_origin = "llm-extractor"
    elif _split_pdf_meta_authors(pdf_meta_author_str):
        raw_authors = _split_pdf_meta_authors(pdf_meta_author_str)
        authors_origin = "pdf-meta"
    else:
        raw_authors = _extract_authors_from_pdf(text)
        authors_origin = "pdf-text"
    authors_list = [_strip_affiliation_marks(a) for a in raw_authors if a]
    authors_list = [a for a in authors_list if a]   # drop empties

    stem = None
    if title and year and authors_list:
        try:
            stem = derive_stem(authors_list, year, title)
        except Exception:
            stem = None

    authors_str = ", ".join(authors_list) if authors_list else None

    # Publication-status hint inferred from DOI prefix + PDF banner text.
    # Imported lazily to avoid a circular import via promote → phases.
    from ..promote import detect_publication_status as _detect_pub
    publication_status = _detect_pub(text, doi, pdf_meta)

    paper_type = llm_meta["paper_type"] or _detect_paper_type(title, venue, text, doi=doi)

    commentary, page_count = _detect_commentary_shape(pdf_path, doi)

    return {
        "stem": stem,
        "title": title,
        "year": year,
        "doi": doi,
        "venue": venue,
        "authors": authors_str,
        "authors_origin": authors_origin,
        "publication_status": publication_status,
        "paper_type": paper_type,
        # Commentary guard (see ..commentary). `page_type` is what the
        # frontmatter writer emits for `type:`; `commentary_signals` is the
        # observable record the promotion gate quotes back. Both are computed
        # once here so no downstream phase re-derives (or disagrees about) them.
        "page_type": commentary.page_type or "paper",
        "commentary_signals": commentary.signals,
        "page_count": page_count,
        "pdf_text_preview": text[:4000],
        "sources": sources,
        "prior_stem": prior_stem,
    }


# ---------- commentary guard ----------

def _doi_via_title_search(
    title: str, first_author: str, year: int | None
) -> str | None:
    """A paper's own DOI, found by matching its title on Semantic Scholar.

    Returns None on any failure — no provider, no hit, or a hit that doesn't
    pass `metadata_sanity`. Never raises: a DOI is a nice-to-have at reconcile
    time, and an S2 outage must not fail an otherwise-good ingest.
    """
    if not first_author:
        # `sanity_ok` would reject everything anyway; skip the request.
        return None
    try:
        from ...providers.semantic_scholar import SemanticScholarProvider
        art = SemanticScholarProvider().search_by_title(title)
    except Exception:
        return None
    if not art:
        return None
    authors = [a.get("name", "") if isinstance(a, dict) else str(a)
               for a in (getattr(art, "authors", None) or [])]
    if not metadata_sanity.sanity_ok(
        getattr(art, "year", None), authors, getattr(art, "title", "") or "",
        first_author, year, title,
    ):
        log(f"s2-title-match ✗ {getattr(art, 'title', '')[:60]!r} "
            f"failed the same-paper check", tag="reconcile")
        return None
    ext = getattr(art, "external_ids", None) or {}
    arxiv = ext.get("ArXiv") if isinstance(ext, dict) else None
    if arxiv:
        # Canonical arXiv namespace beats a publisher aggregator DOI, matching
        # `backfill doi`'s preference.
        return f"10.48550/arXiv.{arxiv}"
    doi = (getattr(art, "doi", None) or "").strip()
    return doi or None


def _detect_commentary_shape(pdf_path: Path, doi: str | None):
    """Run the commentary guard. Returns `(CommentaryVerdict, page_count)`.

    Spends a Crossref request only when a local pre-trigger already fired.

    The two-step shape is the whole point: `crossref_lookup_worthwhile` answers
    False for an ordinary multi-page article, so the common ingest path adds no
    network call at all. When it answers True (single-page document, or page 1
    carries a section label) we complete the signal set with Crossref's
    `reference-count` and `page` — the only fields that make the structural
    tiers decidable, and both already whitelisted for `ingest`.

    The DOI is passed through separately and costs nothing: the news-namespace
    tier reads it directly, so a `10.1038/d…` commentary is caught even when the
    pre-trigger declines to spend a request.
    """
    from ..commentary import crossref_lookup_worthwhile, detect_commentary

    page_count, first_page_text = pdf_shape(pdf_path)
    worthwhile = crossref_lookup_worthwhile(first_page_text, page_count)
    cr = (
        crossref_structural_signals(doi, allow_fetch=True)
        if (doi and worthwhile)
        else {"reference_count": None, "page": None, "type": None, "subtype": None}
    )
    verdict = detect_commentary(
        first_page_text=first_page_text,
        page_count=page_count,
        reference_count=cr["reference_count"],
        crossref_page=cr["page"],
        doi=doi,
    )
    if verdict.is_commentary:
        log(
            f"commentary ⚠ {pdf_path.name} looks like commentary, not a paper "
            f"— signals: {', '.join(verdict.signals)}; "
            f"crossref type={cr['type']!r} subtype={cr['subtype']!r} "
            f"(non-discriminating by design) → suggested type: "
            f"{verdict.page_type}", tag="reconcile"
        )
    elif verdict.considered:
        log(
            f"commentary → not commentary; saw insufficient signal(s): "
            f"{', '.join(verdict.considered)}", tag="reconcile"
        )
    return verdict, page_count


# ---------- pdf-meta cleanup helpers ----------

# pypdf metadata fields are often left at template defaults by PDF
# generators (LaTeX template "Title", Word "Document1", etc). Trusting
# these as-is propagates the placeholder into the wiki — we saw this
# turn into stem `author-2025-title` for an A-Mem paper whose pdfLaTeX
# config had `\title{Title}` left from a template.
_META_TITLE_PLACEHOLDERS = frozenset({
    "title", "untitled", "untitled document", "document", "document1",
    "doc1", "new document", "manuscript", "main", "paper", "article",
    "preprint", "draft", "(no title)", "no title",
})
_META_AUTHOR_PLACEHOLDERS = frozenset({
    "author", "authors", "anonymous", "unknown", "unknown author",
    "user", "owner", "administrator", "admin", "default",
})


def _clean_pdf_meta_value(raw: object, *, kind: str) -> str | None:
    """Return a pypdf metadata string only if it isn't a known placeholder.

    `kind` selects which deny-list to apply: 'title' or 'author'.
    Returns None when the value is empty, whitespace-only, or matches a
    well-known default that PDF generators leave behind.
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    deny = _META_TITLE_PLACEHOLDERS if kind == "title" else _META_AUTHOR_PLACEHOLDERS
    if s.lower() in deny:
        return None
    if s.lower().startswith("microsoft word - "):
        return None
    # Shape check for titles only. The author deny-list is a different problem and
    # author strings legitimately fail a "looks like prose" test.
    if kind == "title" and not _looks_like_title(s):
        return None
    return s


def _split_pdf_meta_authors(raw: object) -> list[str]:
    """Parse pypdf's /Author metadata field into a list of names.

    Common shapes seen in the wild:
      - "Alice; Bob; Carol"          (semicolon — arXiv, ACM)
      - "Alice, Bob, Carol"          (comma — many publishers)
      - "Alice and Bob and Carol"    (BibTeX-style, rare)
      - "Alice Bob"                  (single author, no separator)

    Returns [] if `raw` is empty / non-string / whitespace.
    """
    if not raw or not isinstance(raw, str):
        return []
    s = raw.strip()
    if not s:
        return []
    for sep in (";", " and ", ","):
        if sep in s:
            parts = [p.strip() for p in s.split(sep)]
            return [p for p in parts if p]
    return [s]


def _title_similarity(a: str, b: str) -> float:
    """Token-set Jaccard over normalized title words. Robust to em-dash/hyphen
    differences and word-order tweaks; meaningful only when both strings carry
    real titles (>= 4 content tokens each). Returns 0.0 when either is too
    short to compare, which trips the mismatch path conservatively."""
    def _toks(s: str) -> set[str]:
        s = (s or "").lower()
        s = re.sub(r"[—–\-:;,.()\[\]{}/\"']+", " ", s)
        return {t for t in s.split() if len(t) >= 3}

    ta, tb = _toks(a), _toks(b)
    if len(ta) < 4 or len(tb) < 4:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _detect_paper_type(
    title: str | None,
    venue: str | None,
    text: str,
    doi: str | None = None,
) -> str:
    """Heuristic: classify paper-type for author-prompt selection.

    Returns one of: 'review', 'clinical-trial', 'research' (default).

    Order of evidence (most reliable first):
      1. DOI prefix matches a known short-form section
      2. Trial registration anchors (NCT IDs, ClinicalTrials.gov, EudraCT) →
         'clinical-trial'. Strongest signal — these IDs aren't accidentally
         present in non-trial papers
      3. Title contains 'review', 'survey', or 'perspective' as a word →
         'review'
      4. Title or body has explicit phase 1/2/3 trial language →
         'clinical-trial'
      5. Venue is a known reviews journal → 'review'
      6. Body has a 'this review' / 'in this review' anchor in the first
         5000 chars → 'review'

    Reviews systematically score lower on the semantic-fidelity signal
    because their claims summarize other papers, so the source PDF only
    loosely matches them. The promote gate uses this field to apply a
    relaxed threshold. Clinical-trial detection routes the author to
    `author-system-clinical-trial.md`, which mandates trial structure
    (cohort, primary endpoint, engraftment metrics, AE summary).
    """
    d = (doi or "").lower()
    if d.startswith("10.1056/nejmp") or d.startswith("10.1056/nejmc"):
        return "review"

    head = text[:5000]
    head_lc = head.lower()
    # NCT IDs and other trial-registration anchors are near-binary indicators.
    # NCT format: "NCT" + 8 digits. EudraCT: 4-digit-6-digit-2-digit.
    if re.search(r"\bNCT\d{8}\b", head):
        return "clinical-trial"
    if "clinicaltrials.gov" in head_lc:
        return "clinical-trial"
    if re.search(r"\b\d{4}-\d{6}-\d{2}\b", head):  # EudraCT
        return "clinical-trial"

    t = (title or "").lower()
    if re.search(r"\b(reviews?|surveys?|perspectives?)\b", t):
        return "review"

    # Phase I/II/III trial language in title or head.
    if re.search(r"\bphase\s+(1|2|3|i{1,3})\s+(clinical\s+)?(trial|study)\b", t):
        return "clinical-trial"
    if re.search(r"\bphase\s+(1|2|3|i{1,3})\s+(clinical\s+)?(trial|study)\b", head_lc):
        return "clinical-trial"

    v = (venue or "").lower()
    review_venues = (
        "nature reviews", "annual review", "trends in", "current opinion",
        "chemical reviews", "wires", "annual reviews",
    )
    if any(rv in v for rv in review_venues):
        return "review"

    if re.search(r"\b(in this review|this review (surveys|covers|discusses))\b", head_lc):
        return "review"

    return "research"


# ---------- title / year / author body-text helpers ----------

_BANNED_TITLE_PREFIXES = (
    "nature", "science", "https://", "http://", "doi:", "this is a pdf",
    "accelerated", "received:", "accepted:", "published online", "cite this",
    "arxiv:", "preprint", "© ",
)

# Page-range artifact from production tooling: "3094..3100".
_PAGE_RANGE_RE = re.compile(r"\d+\.\.\d+")

# Journal masthead / running-head signatures. None of these can occur inside a
# real title: a volume-or-issue citation, an explicit page range, or an inline DOI.
_MASTHEAD_LINE_RE = re.compile(
    r"(?:\bvol\.?\s*\d+|\bno\.?\s*\d+\s*,|\bpages?\s+\d+\s*[–—-]\s*\d+"
    r"|\bdoi:\s*\S|\b10\.\d{4,9}/\S"
    # "Advance Access publication January 24, 2014" — Oxford's running head, which
    # trails the section name so no prefix rule reaches it.
    r"|advance access publication"
    # Bare volume:page citation, e.g. "Genome Res. 2009 19: 1655-1664".
    r"|\b\d+\s*:\s*\d+\s*[–—-]\s*\d+)",
    re.IGNORECASE,
)


def _looks_like_title(s: str | None) -> bool:
    """True when `s` plausibly *is* a title rather than production furniture.

    A deny-list of prefixes cannot cover this, because the failure is a shape, not
    a known string. Oxford University Press stamps `/Title` with its internal job
    code and the page range — minimap2's PDF carries
    `OP-CBIO180195 3094..3100`, which is 24 characters and starts with nothing
    banned, so the length-and-prefix check adopted it and short-circuited the
    first-page text scan that would have found the real title. That string then
    became the S2 title-match query (three 404s, retried) and the fallback page
    title. CLAUDE.md is explicit that the naming fields come from "the PDF's first
    page text, not `reader.metadata`", so falling through to the scan is also the
    documented behaviour. Observed 2026-08-10.

    The discriminator is word shape: real titles contain at least three tokens that
    look like words — two or more characters, alphabetic, carrying a lowercase
    letter. Job codes, page ranges, filenames and all-caps mastheads have none or
    almost none. An ALL-CAPS genuine title fails this and falls through to the text
    scan, which is a fallback rather than a failure.
    """
    if not s:
        return False
    s = s.strip()
    if _PAGE_RANGE_RE.search(s):
        return False
    wordy = 0
    for tok in re.split(r"[\s,;:.()\[\]]+", s):
        if len(tok) >= 2 and any(c.isalpha() for c in tok) and any(c.islower() for c in tok):
            wordy += 1
    return wordy >= 3
_AFFILIATION_SUPERSCRIPT_RE: re.Pattern | None = None      # initialized lazily below


def _extract_title_from_pdf(pdf_meta: dict, text: str) -> str | None:
    """Title from PDF metadata when present and reasonable, else multi-line text scan.

    The text scan was previously truncating at the first newline — most journal
    typesetting wraps long titles across two or more lines. We now greedily
    join continuation lines until we hit a blank line, an obviously-author
    line (lots of commas + digits), or 250 chars total.
    """
    meta_title = (pdf_meta.get("/Title") or "").strip()
    if (20 < len(meta_title) < 250
            and not meta_title.lower().startswith(_BANNED_TITLE_PREFIXES)
            and _looks_like_title(meta_title)):
        return meta_title

    head = text[:2000]
    lines = [ln.strip() for ln in head.splitlines()]
    n = len(lines)
    for i, line in enumerate(lines):
        if not (20 < len(line) < 250):
            continue
        if line.lower().startswith(_BANNED_TITLE_PREFIXES):
            continue
        # Masthead, not title. Rejecting a furniture `/Title` above means these
        # PDFs now reach this scan, and for several of them the first qualifying
        # line is the journal masthead rather than the title — bae-2014's is
        # "Vol. 30 no. 10 2014, pages 1473–1475 BIOINFORMATICS APPLICATIONS NOTE
        # doi:10.1093/bioinformatics/btu048", which sails past the prefix and
        # authorish checks. A volume/page citation or an inline DOI is never part
        # of a title, so skip the line and keep scanning for the real one.
        if _MASTHEAD_LINE_RE.search(line):
            continue
        # Cheap heuristic: title lines typically have few commas and few digits;
        # author lines have many. Skip lines that look authorish.
        if line.count(",") >= 2 and any(c.isdigit() for c in line):
            continue
        merged = line
        j = i + 1
        while j < n and j - i <= 3:
            cont = lines[j]
            if not cont:
                break
            if cont.count(",") >= 2 or cont.lower().startswith(_BANNED_TITLE_PREFIXES):
                break
            if cont.endswith(".") or cont.lower().startswith(("abstract", "introduction")):
                break
            if len(merged) + 1 + len(cont) > 250:
                break
            merged = f"{merged} {cont}"
            j += 1
        return merged
    return None


def _extract_year_from_pdf(text: str, doi: str | None = None) -> int | None:
    """Year extraction. Tries (in order):
      1. arXiv DOI YYMM convention.
      2. bioRxiv / medRxiv DOI YYYY.MM.DD convention.
      3. Nature-style 'Received/Accepted/Published <Month> <Year>' header.
      4. 'Cite this article as: ... (YEAR)' line.
      5. bioRxiv watermark 'this version posted <Month> <Day>, <Year>'.
      6. First plausible 4-digit year in the first ~6000 chars (last resort —
         this fallback is the source of the historical reconcile bug where
         a citation in the references list steals the first match before the
         bioRxiv/journal stamps are reached).
    """
    if doi:
        # arXiv post-2007 IDs encode date as YYMM; the DOI form is
        #   10.48550/arXiv.YYMM.NNNNN[vN]   (e.g. 2604.05018 → 2026-04)
        m = re.match(r"10\.48550/arXiv\.(\d{2})(\d{2})\.\d", doi, re.IGNORECASE)
        if m:
            yy = int(m.group(1))
            return 2000 + yy if yy < 60 else None

        # bioRxiv (10.1101/YYYY.MM.DD.NNNNNN) and medRxiv (same scheme).
        # Also covers the newer 10.64898/ prefix bioRxiv migrated to in late
        # 2025 — same path-date encoding. Authoritative; preprint posting
        # year is precisely what the wiki wants.
        m = re.match(r"10\.(?:1101|64898|31219|20944)/(\d{4})\.\d{2}\.\d{2}\.\d",
                     doi, re.IGNORECASE)
        if m:
            yy = int(m.group(1))
            if 2000 <= yy <= 2030:
                return yy

    head = text[:6000]
    # Non-greedy `{0,80}?` picks the FIRST 4-digit year after the keyword
    # within 80 chars — not the last. With greedy matching, a header like
    # "Published online 12 March 2024. Cited Smith et al. 2018." would
    # return 2018 (because `[^\n]{0,80}` swallows up to 2018, then the
    # capture lands on the closest year, which here is 2018).
    for keyword in ("Accepted:", "Published:", "Published online", "Cite this article"):
        for m in re.finditer(rf"{keyword}[^\n]{{0,80}}?\b(20[0-2]\d)\b", head):
            return int(m.group(1))
    # bioRxiv / medRxiv watermark: "this version posted <Month> <Day>, <Year>".
    # Must come BEFORE the bare-year fallback because the references list
    # typically contains older citation years (e.g. Rautiainen 2020) that
    # would steal the first match — which is exactly how Theseus 2026 got
    # mis-stem'd as 2020.
    for m in re.finditer(r"(?:this\s+version\s+)?posted[^\n]{0,40}?\b(20[0-2]\d)\b",
                         head, re.IGNORECASE):
        return int(m.group(1))
    for m in re.finditer(r"\b(20[0-2]\d)\b", head):
        return int(m.group(1))
    return None


def _strip_affiliation_marks(name: str) -> str:
    """Remove trailing affiliation superscripts (digits, asterisks, daggers)
    from an author name. 'Yiwen Song1' → 'Yiwen Song'; 'A. B. White†‡' → 'A. B. White'.

    The character class also covers Unicode asterisk variants
    (∗ U+2217 ASTERISK OPERATOR, ⁎ U+204E LOW ASTERISK) which appear in
    some bioRxiv PDFs in place of the ASCII `*`. Without these, a name
    like `Saro Passaro ∗` survives the strip with `∗` attached, which
    downstream `derive_stem` can reject and substitute to `unknown` —
    the failure mode that produced the `unknown-2024-...` Boltz-2
    duplicate during a 2026-05-27 batch ingest.
    """
    global _AFFILIATION_SUPERSCRIPT_RE
    if _AFFILIATION_SUPERSCRIPT_RE is None:
        _AFFILIATION_SUPERSCRIPT_RE = re.compile(r"\s*[\d\*∗⁎†‡§¶,\s]+$")
    return _AFFILIATION_SUPERSCRIPT_RE.sub("", name).strip()


def _split_author_block(block: str) -> list[str]:
    """Split a comma/and-separated author string into individual cleaned names.

    'Yiwen Song1, Yale Song1, Tomas Pfister1 and Jinsung Yoon1'
        → ['Yiwen Song', 'Yale Song', 'Tomas Pfister', 'Jinsung Yoon']
    """
    if not block:
        return []
    s = block.replace(" and ", ", ")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        cleaned = _strip_affiliation_marks(p)
        if cleaned and 1 < len(cleaned) < 100:
            out.append(cleaned)
    return out


_SENTENCE_STARTERS = (
    "However", "Although", "Despite", "While", "Whereas",
    "In particular", "Furthermore", "Moreover", "Nevertheless",
    "In contrast", "For example", "Specifically", "Therefore",
    "Recently", "Prior", "Existing", "Indeed",
)
_PROSE_STOP_WORDS = frozenset({
    "the", "of", "a", "an", "and", "or", "but", "in", "on",
    "with", "for", "to", "from", "is", "are", "this", "that",
    "these", "those", "we", "our", "their", "its", "by", "as",
    "at", "be", "been", "have", "has", "had", "was", "were",
})


def _looks_like_sentence(line: str) -> bool:
    """Heuristic: does this line look like prose rather than an author list?

    Author lists are mostly proper nouns. Sentences contain stop words,
    end with terminal punctuation, or start with conjunctions like "However,".
    Two-out-of-three of these signals → call it prose, skip it.
    """
    s = line.strip()
    if not s:
        return False
    # Conjunction starts immediately followed by a comma — strong sentence signal.
    for w in _SENTENCE_STARTERS:
        if s.startswith(w + ",") or s.startswith(w + " "):
            return True
    # Sentence-terminating punctuation at line end (allow "et al." / abbreviations).
    if s.endswith((".", ":", "?", "!")) and not s.lower().endswith(("et al.", "et al")):
        return True
    # Stop-word density: author blocks rarely have ≥2 of these.
    tokens = [t.strip(",.;:()[]") for t in s.lower().split()]
    if sum(1 for t in tokens if t in _PROSE_STOP_WORDS) >= 2:
        return True
    return False


# A "name" is 1–4 capitalized words, optionally with a single-letter middle initial.
_NAME_PATTERN = re.compile(
    r"^[A-Z][a-zA-Z'.\-]+(?:\s+(?:[A-Z][a-zA-Z'.\-]+|[A-Z]\.))*$"
)
# Affiliation markers: digit cluster (with optional symbols) OR bare symbol cluster.
_AFFIL_MARKER_RE = re.compile(r"\d+[†‡*§¶]*|[†‡*§¶]+")


def _try_split_by_affiliation_markers(line: str) -> list[str]:
    """Parse 'Edge1† Trinh1† Cheng2 ...' shape author lines.

    Returns the list of names, or [] if the line doesn't look like this
    pattern — meaning the caller should fall through to other strategies.

    Markers (digit clusters + symbol clusters) act as separators. Each chunk
    BEFORE a marker is a candidate name; we accept those matching the
    `_NAME_PATTERN` regex and discard the rest.
    """
    markers = list(_AFFIL_MARKER_RE.finditer(line))
    if len(markers) < 2:
        return []
    chunks: list[str] = []
    last_end = 0
    for m in markers:
        chunk = line[last_end:m.start()].strip(" ,;")
        if chunk:
            chunks.append(chunk)
        last_end = m.end()
    tail = line[last_end:].strip(" ,;")
    if tail:
        chunks.append(tail)

    valid: list[str] = []
    for n in chunks:
        n = n.strip()
        if 3 < len(n) < 80 and _NAME_PATTERN.match(n):
            valid.append(n)
    return valid


def _extract_authors_from_pdf(text: str) -> list[str]:
    """Fallback author-block extraction → list of cleaned names.

    Tries multiple parsing strategies. Sentence-shaped lines (e.g.,
    "However, the use of RAG ...") are skipped in both — the prior
    implementation got fooled by such lines because they have commas
    + uppercase + reasonable length, the same surface features as an
    author list.

    Strategies (in order; first non-empty result wins):
      1. Comma-separated authors:
           "Y. Du, X. Tang, et al."
      2. Space-separated names with superscript affiliation markers:
           "Darren Edge1† Ha Trinh1† Newman Cheng2 ..."
         The marker is the separator (no commas needed).

    Returns [] when no plausible author block found — caller falls
    through to whatever's next in the reconcile chain.
    """
    head = text[:1500]
    lines = head.splitlines()

    # Strategy 1: comma-separated.
    for line in lines:
        s = line.strip()
        if "," in s and 20 < len(s) < 500 and any(c.isupper() for c in s):
            if _looks_like_sentence(s):
                continue
            cleaned = _split_author_block(s)
            if cleaned:
                return cleaned

    # Strategy 2: space-separated with affiliation markers (the GraphRAG case).
    for line in lines:
        s = line.strip()
        if 20 < len(s) < 500 and not _looks_like_sentence(s):
            names = _try_split_by_affiliation_markers(s)
            if len(names) >= 2:
                return names

    return []
