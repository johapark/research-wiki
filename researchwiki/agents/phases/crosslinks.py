"""Cross-link discovery + verification phases.

`crosslink_candidates` runs early in ingest to seed the author with a
*verified* whitelist of [[wikilinks]] sourced from the citation graph
(S2 references / citations, Crossref refs, PDF DOI scan).

`propose_crosslinks` runs alongside it: semantic-KNN against the wiki page
index nominates candidates the citation graph misses, then a strict LLM judge
keeps only source-supported engagement — the supplied paper excerpts must
explicitly build on or contrast the candidate. Shared topic/problem alone is
not linkable under the repository's cross-link contract.

`verify_crosslinks` runs at commit time to strip any [[wikilink]] the author
wrote that wasn't on the whitelist or whose target page is missing — the
combined effect is that the author has no incentive to invent links.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ...paths import wiki_dir
from ...pdf.text import extract_ref_dois
from ...providers.semantic_scholar import SemanticScholarProvider
from ...index import pages_semantic as semantic_pages
from ...wiki import read_page, read_wiki_dois
from .. import llm
from ...errors import EnvironmentFailure
from ...log import log


_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


@dataclass
class CrosslinkCandidate:
    """A wiki page that references the source paper or is referenced by it.

    The candidate's `kind` records the discovered direction:
      - 'cited_by_source'   — source paper cites this wiki page (PDF-ref or S2 /references)
      - 'cites_source'      — wiki page cites the source paper (S2 /citations)
      - 'topical'           — legacy token for an LLM-confirmed, source-supported
                              build-on/contrast relationship

    `relationship` is populated only for `topical` candidates — a one-line
    LLM-provided rationale that the author phase can use as the bullet text
    in the Related Papers section. Citation-graph candidates leave it empty;
    the author phase generates its own one-liner from paper context.

    Only candidates with `verified=True` are passed to the author phase as
    safe-to-use [[wikilinks]]. Here "verified" means citation-graph evidence or
    a strict source-engagement judgment over PDF-grounded excerpts — never
    semantic similarity alone. The commit phase strips any wikilink the author
    writes that isn't in the verified candidate list.
    """
    doi: str
    wikilink: str           # 'category/stem' (no [[]] wrapper)
    kind: str
    title: str
    year: int | str | None
    verified: bool          # True iff citation-backed or source-engagement judged
    relationship: str = ""  # LLM-provided one-liner for `topical` candidates


@dataclass
class VerificationReport:
    """Outcome of cross-link verification on a draft."""
    verified: list[str]      # wikilinks kept (in candidate list AND target page exists)
    unverified: list[str]    # wikilinks stripped (not in candidate list)
    broken: list[str]        # wikilinks where the target wiki/{key}.md doesn't exist


def crosslink_candidates(
    pdf_path: Path,
    metadata: dict,
    *,
    stats: dict | None = None,
) -> list[CrosslinkCandidate]:
    """Find wiki pages plausibly cross-linkable to the source paper.

    Sources, in order of trust:
      1. S2 /references → wiki DOIs    (source paper cites our wiki page)
      2. S2 /citations  → wiki DOIs    (wiki page cites the source paper)
      3. PDF-text DOI scan → wiki DOIs (fallback when S2 has no record)

    Returns at most ~20 verified candidates. The author phase gets these
    as a whitelist; the commit-phase verifier strips any wikilink not in
    this list.

    `stats`, when passed, is filled with diagnostics for the caller. The one
    that matters is `citation_graph_unresolved`: S2 knows the paper and reports
    a nonzero `referenceCount`, but `/references` came back empty. That happens
    for freshly-registered DOIs whose reference list S2 has not resolved yet,
    and it is NOT the same as "this paper cites nothing in the wiki" — there is
    simply no citation evidence either way. Observed 2026-08-04 on a Nature
    Reviews Genetics review: referenceCount 139, `/references` data 0.
    """
    wiki_dois = read_wiki_dois()                  # {doi_lower: 'category/stem'}
    if stats is not None:
        stats.setdefault("citation_graph_unresolved", False)
    if not wiki_dois:
        return []

    candidates: list[CrosslinkCandidate] = []
    seen_dois: set[str] = set()
    doi = metadata.get("doi")

    if doi:
        provider = SemanticScholarProvider()
        article = provider.get_by_doi(doi)

        if article is not None:
            refs = provider.get_references(article)
            if not refs and (getattr(article, "reference_count", 0) or 0) > 0:
                if stats is not None:
                    stats["citation_graph_unresolved"] = True
                log(
                    f"S2 reports {article.reference_count} references but "
                    f"/references returned none — citation graph unresolved for "
                    f"this DOI; cross-links have no citation evidence",
                    tag="propose_crosslinks",
                )
            for ref in refs:
                ref_doi = (ref.doi or "").lower()
                if ref_doi and ref_doi in wiki_dois and ref_doi not in seen_dois:
                    seen_dois.add(ref_doi)
                    candidates.append(CrosslinkCandidate(
                        doi=ref_doi,
                        wikilink=wiki_dois[ref_doi],
                        kind="cited_by_source",
                        title=ref.title or "",
                        year=ref.year,
                        verified=True,
                    ))
            for cit in provider.get_citations(article):
                cit_doi = (cit.doi or "").lower()
                if cit_doi and cit_doi in wiki_dois and cit_doi not in seen_dois:
                    seen_dois.add(cit_doi)
                    candidates.append(CrosslinkCandidate(
                        doi=cit_doi,
                        wikilink=wiki_dois[cit_doi],
                        kind="cites_source",
                        title=cit.title or "",
                        year=cit.year,
                        verified=True,
                    ))

    # Crossref fallback — useful for Nature-style papers where pypdf can't
    # parse the reference list cleanly. Crossref /works returns the publisher-
    # deposited reference DOIs even when S2 doesn't have the paper.
    if doi and (not candidates or len(candidates) < 5):
        from ...providers.crossref import fetch_crossref_refs
        cr_dois = fetch_crossref_refs(doi)
        for ref_doi in cr_dois:
            ref_doi_l = ref_doi.lower()
            if ref_doi_l in wiki_dois and ref_doi_l not in seen_dois:
                seen_dois.add(ref_doi_l)
                candidates.append(CrosslinkCandidate(
                    doi=ref_doi_l,
                    wikilink=wiki_dois[ref_doi_l],
                    kind="cited_by_source",
                    title="",
                    year=None,
                    verified=True,
                ))

    # PDF-text DOI scan — last resort, useful for preprints lacking both
    # S2 and Crossref records.
    if not candidates or len(candidates) < 5:
        try:
            pdf_dois = extract_ref_dois(pdf_path, own_doi=doi)
        except Exception:
            pdf_dois = []
        for ref_doi in pdf_dois:
            ref_doi_l = ref_doi.lower()
            if ref_doi_l in wiki_dois and ref_doi_l not in seen_dois:
                seen_dois.add(ref_doi_l)
                candidates.append(CrosslinkCandidate(
                    doi=ref_doi_l,
                    wikilink=wiki_dois[ref_doi_l],
                    kind="cited_by_source",
                    title="",
                    year=None,
                    verified=True,
                ))

    return candidates[:20]


def verify_crosslinks(
    draft_text: str,
    candidates: list[CrosslinkCandidate],
) -> tuple[str, VerificationReport]:
    """Strip unverified [[wikilinks]] from a draft.

    A wikilink is kept only if BOTH:
      1. It appears in the verified `candidates` list (S2/Crossref-confirmed
         relationship between the source paper and the target wiki page).
      2. The target wiki page actually exists at wiki/{category}/{stem}.md.

    Unverified or broken wikilinks are stripped, keeping any surrounding
    prose intact. The report documents what was removed for the commit
    decision_reason field — provenance preserved even when the link isn't.
    """
    candidate_keys = {c.wikilink for c in candidates}
    wiki_root = wiki_dir()

    verified: list[str] = []
    unverified: list[str] = []
    broken: list[str] = []

    def _replacer(m: re.Match) -> str:
        key = m.group(1).strip()
        target = wiki_root / f"{key}.md"
        exists = target.exists()
        if key in candidate_keys and exists:
            verified.append(key)
            return m.group(0)            # keep verbatim
        if not exists:
            broken.append(key)
        else:
            unverified.append(key)
        return ""

    cleaned = _WIKILINK_RE.sub(_replacer, draft_text)

    # Empty-bullet cleanup: if a Related Papers bullet was just a wikilink,
    # the result is a dangling "- " line. Drop those.
    cleaned = re.sub(r"^\s*-\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned, VerificationReport(
        verified=sorted(set(verified)),
        unverified=sorted(set(unverified)),
        broken=sorted(set(broken)),
    )


# ---------- semantic crosslink proposals ----------

# Default neighbor depth for semantic candidate retrieval. Cap is intentionally
# modest: deeper pools waste LLM tokens on noise (cosine drops off fast past
# the top 8) and the author can only use ≤6 wikilinks anyway.
SEMANTIC_K = 8


def propose_crosslinks(
    metadata: dict,
    sections: dict,
    *,
    k: int = SEMANTIC_K,
    use_stub: bool = False,
    exclude_keys: frozenset[str] = frozenset(),
    allow_gleaning: bool = True,
) -> list[CrosslinkCandidate]:
    """Find source-supported crosslink candidates the citation graph misses.

    Pipeline (A-Mem Link Generation analog):
      1. Embed the new paper's title + summary excerpt (semantic page index).
      2. Top-k semantic neighbors among existing paper-type pages.
      3. Single LLM call judges each against source/candidate excerpts. Keep only
         explicit build-on/contrast relationships; shared topic, task, or
         vocabulary is `none`.

    Returns CrosslinkCandidate list with `kind="topical"`, `verified=True`,
    and `relationship` populated. Caller unions these with the citation-graph
    candidates from `crosslink_candidates`.

    `exclude_keys` lets the caller drop wikilinks already present in the
    citation-graph candidate list — no point asking the LLM to judge them
    again, and we avoid duplicate entries reaching the author.

    `allow_gleaning=False` disables the second-chance pass. Pass it when the
    citation graph came back unresolved (see `crosslink_candidates`'s `stats`):
    without structural evidence, re-opening candidates rejected in pass 1
    adds model pressure without adding source evidence.
    """
    if not semantic_pages.index_exists():
        return []

    title = (metadata.get("title") or "").strip()
    preview = (metadata.get("pdf_text_preview") or "")[:1500]
    summary_block = (sections or {}).get("results", "")[:600]
    probe = "\n".join(filter(None, [title, preview, summary_block])).strip()
    if not probe:
        return []

    # Restrict to paper-type pages — synthesis pages have a different
    # cross-link contract and are handled by `propose_evolution`.
    hits = semantic_pages.query_text(
        probe,
        k=k * 2,                     # over-fetch; we'll filter by exclude_keys
        page_types=("paper",),
    )
    eligible = [h for h in hits if h.key not in exclude_keys]
    existing = [h for h in eligible if _candidate_page(h) is not None]
    n_stale = len(eligible) - len(existing)
    hits = existing[:k]
    if n_stale:
        log(
            f"dropped {n_stale} stale/missing semantic candidate(s)",
            tag="propose_crosslinks",
        )
    if not hits:
        return []

    if use_stub:
        # Similarity only nominates candidates; the stub cannot perform the
        # source-engagement judgment required to turn one into a wikilink.
        return []

    judged = _judge_candidates(
        metadata, sections, hits, allow_gleaning=allow_gleaning
    )
    return judged


def _judge_candidates(
    metadata: dict,
    sections: dict,
    hits: list,
    *,
    allow_gleaning: bool = True,
) -> list[CrosslinkCandidate]:
    """Two-pass LLM judge for a batch of semantic candidates.

    Pass 1 — strict filter. Batched because per-candidate calls cost N× the
    latency and lose the comparative context an LLM benefits from when ranking
    related work. A 1.5K-token prompt covering 8 candidates beats 8 separate
    400-token calls.

    Pass 2 — one strict recall pass. When pass 1 kept ≤2 and rejected ≥3,
    re-check whether it overlooked explicit build-on/contrast evidence. The
    acceptance bar does not move: speculative or merely adjacent candidates
    remain unlinked. Capped at one round so the call count stays bounded.
    """
    block = _build_judge_prompt(metadata, sections, hits)
    try:
        resp = llm.call(
            phase="link_generation",
            prompt=block,
            system=_JUDGE_SYSTEM,
            use_stub=False,
            schema=_JUDGE_SCHEMA,
        )
    except EnvironmentFailure:
        raise  # house rule 1 (errors.py): never absorb a provider outage
    except Exception as e:
        log(f"judge call failed: {e}", tag="propose_crosslinks")
        return []

    verdicts = _parse_judge_response(resp.text)
    by_key = {h.key: h for h in hits}

    out: list[CrosslinkCandidate] = []
    promoted_keys: set[str] = set()
    rejected_keys: list[str] = []
    for v in verdicts:
        key = v.get("wikilink")
        verdict_kind = (v.get("verdict") or "").strip().lower()
        if key not in by_key:
            continue
        if verdict_kind != "topical":
            rejected_keys.append(key)
            continue
        h = by_key[key]
        out.append(CrosslinkCandidate(
            doi="",
            wikilink=key,
            kind="topical",
            title=h.title,
            year=None,
            verified=True,
            relationship=(v.get("rationale") or "").strip()[:200],
        ))
        promoted_keys.add(key)

    # Gleaning: when recall looks low, give the LLM one more chance to find
    # explicit engagement evidence it overlooked. The second pass has the same
    # precision bar; it cannot promote borderline adjacency.
    if not allow_gleaning:
        log(f"gleaning suppressed (citation graph unresolved; pass-1: "
            f"{len(out)} topical, {len(rejected_keys)} rejected)",
            tag="propose_crosslinks")
    elif len(out) <= 2 and len(rejected_keys) >= 3:
        log(f"gleaning fires (pass-1: {len(out)} topical, "
              f"{len(rejected_keys)} rejected)", tag="propose_crosslinks")
        gleaned = _gleaning_pass(metadata, sections, hits, rejected_keys, by_key)
        n_added = 0
        for cand in gleaned:
            if cand.wikilink in promoted_keys:
                continue
            out.append(cand)
            promoted_keys.add(cand.wikilink)
            n_added += 1
        log(f"gleaning → +{n_added} candidate(s)", tag="propose_crosslinks")
    elif len(rejected_keys) >= 3:
        log(f"gleaning skipped (pass-1: {len(out)} topical, "
              f"{len(rejected_keys)} rejected — precision-bound)", tag="propose_crosslinks")

    return out


def _gleaning_pass(
    metadata: dict,
    sections: dict,
    hits: list,
    rejected_keys: list[str],
    by_key: dict,
) -> list[CrosslinkCandidate]:
    """Re-prompt the judge on its rejected candidates only.

    The intent isn't to lower pass 1's bar, but to catch explicit build-on or
    contrast evidence that was overlooked. Both passes authorize a durable
    wikilink, so both must meet the same source-supported threshold.
    """
    if not rejected_keys:
        return []
    rejected_hits = [h for h in hits if h.key in rejected_keys]
    if not rejected_hits:
        return []

    block = _build_gleaning_prompt(metadata, sections, rejected_hits)
    try:
        resp = llm.call(
            phase="link_generation",
            prompt=block,
            system=_GLEANING_SYSTEM,
            use_stub=False,
            schema=_JUDGE_SCHEMA,
        )
    except EnvironmentFailure:
        raise  # house rule 1 (errors.py): never absorb a provider outage
    except Exception as e:
        log(f"gleaning call failed: {e}", tag="propose_crosslinks")
        return []

    out: list[CrosslinkCandidate] = []
    for v in _parse_judge_response(resp.text):
        key = v.get("wikilink")
        verdict_kind = (v.get("verdict") or "").strip().lower()
        if key not in by_key or verdict_kind != "topical":
            continue
        h = by_key[key]
        rationale = (v.get("rationale") or "").strip()[:200]
        # Tag gleaned hits in the rationale so an author skimming knows
        # this came from the recall pass and warrants extra scrutiny.
        rationale = f"(gleaning) {rationale}" if rationale else "(gleaning)"
        out.append(CrosslinkCandidate(
            doi="",
            wikilink=key,
            kind="topical",
            title=h.title,
            year=None,
            verified=True,
            relationship=rationale,
        ))
    return out


# JSON Schema for the crosslink judge envelope (used by both _judge_candidates
# and _gleaning_pass). Honored by chat-relay; ignored by other providers.
# `verdict` is enum-constrained because downstream code matches against
# specific lowercase strings. There is deliberately no "borderline" value:
# speculative relationships may be reported elsewhere, but may not become
# durable Related-Papers wikilinks.
_JUDGE_SCHEMA = {
    "type": "object",
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["wikilink", "verdict"],
                "properties": {
                    "wikilink": {"type": "string"},
                    "verdict":  {"type": "string",
                                 "enum": ["topical", "none"]},
                    "rationale": {"type": ["string", "null"]},
                },
            },
        },
    },
}


_JUDGE_SYSTEM = """\
You are a citation auditor. Given a new paper and a shortlist of existing
wiki pages that semantically resemble it, decide which pairings represent
a real intellectual relationship versus coincidental vocabulary overlap.

For each candidate, output one of:
  - "topical" — compatibility label for an EXPLICIT source-supported
    relationship: the new-paper excerpts show that it builds on or contrasts
    the candidate's named method, result, or claim. Provide a 12-25 word
    rationale naming that evidence.
  - "none" — no explicit engagement is supported. This includes merely solving
    the same problem, using similar methods independently, sharing a field, or
    resembling the candidate only semantically. No rationale needed.

Be strict. False positives waste an author's time and dilute the wiki's
cross-link signal. When in doubt, return "none".

Output JSON only, shape:
{"verdicts": [
  {"wikilink": "category/stem", "verdict": "topical", "rationale": "..."},
  {"wikilink": "category/stem", "verdict": "none"}
]}
"""

_GLEANING_SYSTEM = """\
You are reviewing your own previous verdicts on candidate wiki cross-links.
The shortlist below is the subset you marked "none" on the first pass — i.e.
candidates you rejected as superficial.

The first pass was strict by design (better to under-link than dilute the
wiki). This pass is a recall check: did it overlook explicit evidence in the
provided excerpts that the new paper builds on or contrasts a candidate?

For each candidate, output one of:
  - "topical" — compatibility label for explicit build-on/contrast evidence
    the first pass missed. Name that evidence in the rationale.
  - "none" — the original verdict stands. Plausible, uncertain, same-problem,
    or topically adjacent relationships remain `none` because this output can
    become a durable wikilink.

Most candidates will stay "none". Promote one or two if the case is solid.
False promotions reverse the value of the strict first pass.

Output JSON only, shape:
{"verdicts": [
  {"wikilink": "category/stem", "verdict": "topical", "rationale": "..."},
  {"wikilink": "category/stem", "verdict": "none"}
]}
"""


def _build_gleaning_prompt(metadata: dict, sections: dict, rejected_hits: list) -> str:
    parts = [
        "# New paper",
        f"Title: {metadata.get('title') or 'unknown'}",
        f"Year:  {metadata.get('year') or 'unknown'}",
        "",
        *_source_excerpt_blocks(metadata, sections),
        "# Previously-rejected candidates — re-check for explicit engagement",
    ]
    for h in rejected_hits:
        page = _candidate_page(h)
        if page is None:
            continue
        parts.extend(_format_candidate_block(h, page))
    parts.append("---")
    parts.append("Output JSON per the system prompt. One verdict per "
                 "candidate. No prose outside the JSON.")
    return "\n".join(parts)


def _build_judge_prompt(metadata: dict, sections: dict, hits: list) -> str:
    parts = [
        "# New paper",
        f"Title: {metadata.get('title') or 'unknown'}",
        f"Year:  {metadata.get('year') or 'unknown'}",
        "",
        *_source_excerpt_blocks(metadata, sections),
        "# Candidate wiki pages (top semantic neighbors)",
    ]
    for h in hits:
        page = _candidate_page(h)
        if page is None:
            continue
        parts.extend(_format_candidate_block(h, page))
    parts.append("---")
    parts.append("Output JSON per the system prompt. One verdict per "
                 "candidate. No prose outside the JSON.")
    return "\n".join(parts)


def _source_excerpt_blocks(metadata: dict, sections: dict) -> list[str]:
    """Render the same PDF-grounded context for both judge passes.

    Explicit engagement most often appears in the introduction or discussion,
    while methods and results establish what was actually reused or contrasted.
    Keeping both passes on one helper prevents the recall pass from judging with
    a thinner evidence window than the initial pass.
    """
    sections = sections or {}
    sources = (
        ("Summary excerpt (PDF first page):",
         metadata.get("pdf_text_preview") or "", 1200),
        ("Introduction excerpt:", sections.get("introduction") or "", 800),
        ("Methods excerpt:", sections.get("methods") or "", 600),
        ("Results excerpt:", sections.get("results") or "", 600),
        ("Discussion excerpt:", sections.get("discussion") or "", 800),
    )
    parts: list[str] = []
    for label, text, limit in sources:
        excerpt = text.strip()
        if not excerpt:
            continue
        parts.extend((label, excerpt[:limit], ""))
    return parts


def _candidate_page(hit):
    """Read a semantic hit's canonical page, tolerating stale index rows.

    Page removal and a subsequent clean re-ingest intentionally leave derived
    indexes stale until the new page is promoted. Retrieval may therefore
    return a key whose Markdown file no longer exists. A missing candidate is
    simply ineligible evidence for cross-link judging, not an ingest failure.
    """
    try:
        return read_page(wiki_dir() / f"{hit.key}.md")
    except OSError:
        return None


def _top_graded_claims(stem: str, k: int = 3) -> list[dict]:
    """Return the top-k graded claims for `stem`, ranked by semantic_score.

    Used to inject grounded, citable units into the crosslink judge prompt
    instead of relying on free-prose Summary text. When grading hasn't run
    on the candidate yet, returns whatever claims exist in section/position
    order so the prompt doesn't go empty for un-graded papers.

    Silent no-op (returns []) when the DB is unreachable — the caller falls
    back to the existing Summary-only prompt.
    """
    try:
        from ...tools import claims_by_stem
        rows = claims_by_stem(stem)
    except Exception:
        return []
    if not rows:
        return []
    graded = [r for r in rows if r.get("semantic_score") is not None]
    if graded:
        graded.sort(
            key=lambda r: (
                -(r.get("semantic_score") or 0.0),
                -(r.get("bm25_top1") or 0.0),
            )
        )
        return graded[:k]
    return rows[:k]


def _format_candidate_block(h, page) -> list[str]:
    """Render one cross-link candidate's prompt block.

    Today's shape: `## [[key]]  (cos=…)` + `Title: …` + `Summary: …`. The
    Summary is narrative prose — useful but unscored. We append a
    `Top graded claims:` block when the structured DB has any, so the
    judge sees pre-graded, addressable units (sem-scored) alongside the
    Summary. Falls through to Summary-only when no claims exist.
    """
    cand_summary = _extract_summary(page.body)[:600]
    parts = [
        f"## [[{h.key}]]  (cos={h.score:.2f})",
        f"Title: {page.fm.get('title','')}",
        f"Summary: {cand_summary}",
    ]
    stem = h.key.split("/", 1)[1] if "/" in h.key else h.key
    top_claims = _top_graded_claims(stem, k=3)
    if top_claims:
        parts.append("Top graded claims:")
        for c in top_claims:
            text = (c.get("text") or "").strip()
            if len(text) > 220:
                text = text[:217].rstrip() + "..."
            sem = c.get("semantic_score")
            score_tail = f" (sem={sem:.2f})" if sem is not None else " (ungraded)"
            section = c.get("section", "")
            position = c.get("position", 0)
            parts.append(f"  - [{section}#{position}] {text}{score_tail}")
    parts.append("")
    return parts


def _extract_summary(body: str) -> str:
    """Pull `## Summary` body. Local copy to avoid importing wiki.extract_section
    here and creating a churn-y dependency cycle."""
    m = re.search(r"^##\s+Summary\s*$\s*(.+?)(?=^##\s+|\Z)",
                  body, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else body[:600]


def _parse_judge_response(text: str) -> list[dict]:
    """Tolerate fenced code, leading prose, or stray quotes around the JSON."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    items = obj.get("verdicts") or []
    return [v for v in items if isinstance(v, dict)]
