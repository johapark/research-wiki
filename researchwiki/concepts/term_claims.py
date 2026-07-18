"""Shared helpers over the term ↔ claim ↔ paper substrate.

These functions bridge concept terms and the claim rows in state.db:

  - `_matching_claims`   — find graded claims mentioning a term/keyword
  - `_page_mentions`     — does the term appear in a page's prose?
  - `_term_claim_hint`   — one-line authoring hint pulled from a member's claims
  - `_best_claim_slug`   — canonical claim anchor for `[[stem#slug]]` spokes
  - `_top_kc_claim_slug` — fallback anchor when a member has no direct term-match

Used by both `scaffold` (to build a new hub) and `attach` (to slot a new
paper into existing hubs). Kept in its own module so those two entry
points don't have to import each other.
"""

from __future__ import annotations

import re

from ..log import log
from ..paths import wiki_dir

# Term-slug function is defined in candidates.py — imported to avoid duplication.
from .candidates import _load_paper_metadata, _term_slug

# Section tags whose claims count as a paper's actual contribution (as opposed
# to its context / related work / limitations). Concept-attachment gating.
_CONTRIBUTION_SECTIONS = ("key_contributions", "results", "methodology")


def _promote_instantiates_edges_for(term: str) -> int:
    """When a hub is scaffolded, promote its `instantiates` edges to `confirmed`.

    The scaffold itself IS the review signal — the human decided the term is
    hub-worthy, so every claim previously flagged as instantiating it is
    endorsed en bloc. Returns the number of edges transitioned. Silent no-op
    on any failure.
    """
    try:
        from ..claim_graph import open_edges_db, query, set_status
        term_slug = _term_slug(term)
        if not term_slug:
            return 0
        conn = open_edges_db()
        try:
            edges = query(
                conn, relation="instantiates",
                tgt_stem="concepts", tgt_slug=term_slug, status="candidate",
            )
            n = 0
            for e in edges:
                if set_status(conn, e.id, "confirmed"):
                    n += 1
            conn.commit()
            return n
        finally:
            conn.close()
    except Exception as e:
        log(f"instantiates promotion skipped: {type(e).__name__}: {e}",
            tag="concepts")
        return 0

def _page_mentions(term: str, prose: str) -> bool:
    """True iff `term` occurs in `prose` as a whole, case-sensitive token.

    Membership uses a boundary-anchored match on the *exact* user-supplied term
    rather than reusing lint's discovery tokenizer: the tokenizer greedily
    absorbs a preceding capital ("The Virtual Cell" → one phrase), so exact-token
    equality would miss the term at sentence start. A case-sensitive boundary
    match is robust to position while still rejecting substrings and case-folds
    — "RAG" won't match "storage", "CAR" won't match "car". Boundaries are
    non-alphanumerics, so "CAR-T"/"RAG-based" still match the leading token.
    """
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", prose) is not None

def _matching_claims(stem: str, term: str) -> list[dict]:
    """Return [{section, position, claim_slug, text, semantic_score}] for
    claims in `stem` whose text contains `term` (case-insensitive, at word
    boundaries) AND live in a contribution section (see _CONTRIBUTION_SECTIONS).

    Word-boundary matching stops short acronyms like "DMS" from matching
    inside longer words like "Dmse" (a mean-squared-error variable) or
    "DMSO" (the solvent). SQLite's LIKE does the coarse filter; a Python
    regex applies the boundary check on the returned rows.

    Ordering: kc → results → methodology, then by position within section.
    Result[0] is the best pick for a spoke citation. Empty on any DB error or
    when the paper has no matching claims (paper mentions the term only in
    body prose, which is *not* enough to make it a hub member).
    """
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return []
    try:
        rows = conn.execute(
            "SELECT section, position, claim_slug, text, semantic_score "
            "  FROM claims "
            " WHERE paper_stem = ? "
            "   AND section IN (?, ?, ?) "
            "   AND claim_slug IS NOT NULL "
            "   AND is_cross_ref = 0 "
            "   AND LOWER(text) LIKE ? "
            " ORDER BY CASE section "
            "     WHEN 'key_contributions' THEN 0 "
            "     WHEN 'results' THEN 1 "
            "     WHEN 'methodology' THEN 2 "
            "     ELSE 3 END, position",
            (stem, *_CONTRIBUTION_SECTIONS, f"%{term.lower()}%"),
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()

    # Word-boundary post-filter. Non-alphanumeric neighbors on both sides,
    # or start/end of string. Matches the boundary logic used by
    # `_page_mentions` above — hyphens and underscores don't extend the
    # match, so "off-target" still finds "off-target detection".
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)
    return [
        {
            "section": r["section"], "position": r["position"],
            "claim_slug": r["claim_slug"], "text": r["text"],
            "semantic_score": r["semantic_score"],
        }
        for r in rows if pat.search(r["text"])
    ]

def _term_claim_hint(stem: str, term: str) -> str:
    """A `stem` claim that actually mentions `term`, truncated — an authoring
    hint inlined as an HTML comment (no `claim_id`, per CLAUDE.md).

    Prefers a claim whose text contains the term (that's the one describing how
    the paper *uses* the concept); among those, the highest-scoring. Falls back
    to '' when no claim mentions the term — better an empty hint the author
    fills than a top-scoring claim about something unrelated. '' on any miss.
    """
    try:
        from ..search import claims_by_stem
        rows = claims_by_stem(stem)
    except Exception:
        return ""
    hits = [r for r in rows if term in (r.get("text") or "")]
    if not hits:
        return ""
    graded = [r for r in hits if r.get("semantic_score") is not None]
    pick = max(graded, key=lambda r: r["semantic_score"]) if graded else hits[0]
    text = (pick.get("text") or "").strip().replace("-->", "--")
    return text[:160].rstrip() + ("…" if len(text) > 160 else "")

def _best_claim_slug(stem: str, term: str) -> str | None:
    """Highest-priority contribution-section claim slug for `term` in `stem`.

    Ordering follows _matching_claims (kc > results > methodology > position).
    Returns None when no contribution claim mentions the term — the caller
    falls back to bare `[[stem]]` in that case.
    """
    hits = _matching_claims(stem, term)
    return hits[0]["claim_slug"] if hits else None

def _keyword_matches_term(keyword: str, term: str, term_slug: str) -> bool:
    """Three cheap variant checks that decide whether a paper's LLM-authored
    keyword matches a hub's target term. Order matters — direct equality is
    cheapest, slug-equal is strictest, substring is most permissive."""
    kw = keyword.strip().lower()
    tm = term.strip().lower()
    if not kw or not tm:
        return False
    # (1) direct case-insensitive match
    if kw == tm:
        return True
    # (2) term appears as a substring of the keyword — catches "PAM" ↔
    # "PAM variants", "prime editing" ↔ "prime editing efficiency"
    if tm in kw:
        return True
    # (3) slugified forms agree — catches hyphen ↔ space variants
    # ("off-target" ↔ "off target", "prime-editing" ↔ "prime editing")
    return bool(term_slug) and _term_slug(keyword) == term_slug

def _papers_where_keywords_match(
    term: str, aliases: list[str] | None = None,
) -> dict[str, list[str]]:
    """Return {stem → matched-keywords-list} for papers whose LLM-authored
    keywords or tags contain the term (via any of the three variant checks),
    OR any of the caller-supplied `aliases`.

    Aliases handle the vocabulary-divergence case: "deep mutational scanning"
    (Fanton, Yu) ↔ "saturation mutagenesis" (Zhou, Lawson) ↔ "DMS" (Tabet)
    ↔ "MAVE" — the LLM normalized each paper's terminology, but not across
    papers. The hub author supplies the alias set (`--aliases "a,b,c"` on
    scaffold; `topic_seed_aliases:` in the hub YAML for later hooks).

    The keyword list is retained so a caller can use it as an alias set when
    widening the claim-substring match. Empty dict when state.db is
    unreachable or holds no papers.
    """
    term_lc = term.strip().lower()
    if not term_lc:
        return {}
    # Deduplicate the term+aliases search set. Preserve order for determinism.
    search_terms: list[str] = [term]
    seen = {term_lc}
    for a in aliases or []:
        if not isinstance(a, str):
            continue
        a_lc = a.strip().lower()
        if a_lc and a_lc not in seen:
            search_terms.append(a)
            seen.add(a_lc)
    # Precompute slugs for the slug-equality check.
    slugs = {_term_slug(t): t for t in search_terms if _term_slug(t)}

    out: dict[str, list[str]] = {}
    for paper in _load_paper_metadata():
        matched: list[str] = []
        for source in ("keywords", "tags"):
            for kw in paper.get(source, []) or []:
                if not isinstance(kw, str):
                    continue
                # Check against every search term / alias in turn.
                for t in search_terms:
                    if _keyword_matches_term(kw, t, _term_slug(t)):
                        matched.append(kw)
                        break
        if matched:
            out[paper["stem"]] = matched
    return out

def _top_kc_claim_slug(stem: str) -> str | None:
    """Fallback anchor: the paper's first key_contributions claim slug.

    Used when a paper is a keyword-hit member but no claim (nor any of the
    paper's own keywords used as an alias) matches directly — we still want
    a `[[stem#slug]]` citation, so we point at the paper's top contribution.
    """
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return None
    try:
        row = conn.execute(
            "SELECT claim_slug FROM claims "
            " WHERE paper_stem = ? AND section = 'key_contributions' "
            "   AND claim_slug IS NOT NULL AND is_cross_ref = 0 "
            " ORDER BY position LIMIT 1",
            (stem,),
        ).fetchone()
    except Exception:
        return None
    finally:
        conn.close()
    return row["claim_slug"] if row and row["claim_slug"] else None
