"""Read-only retrieval primitives shared by the `claims` and `pdf-search` CLIs.

These two functions back capabilities that have no other CLI home:

  - `claim_lookup` — grounded-citation search over the pre-graded `claims`
    table (each hit carries a content-addressed `claim_slug` + support
    scores). Cite hits as `[[stem#slug]]` — durable, survives `db rebuild`.
  - `pdf_section_search` — BM25 search inside one paper's PDF chunks.

The other read paths (`wiki_search`, `wiki_get_page`, `db_query`) live as
first-class CLI commands (`researchwiki search`, native file `Read`, and
`researchwiki db query`) — no duplicate wrappers here.

Design rules:
  - No LLM calls. Deterministic.
  - Returns are JSON-serializable (dict / list of dict / list of primitives).
  - Failures return an empty result + a `note` field, not an exception.
"""

from __future__ import annotations

import re
import sqlite3

from ..db.connection import get_connection
from ..index.pdf_chunks import build_pdf_index, query_pdf


# ---------- pdf_section_search ----------

def pdf_section_search(stem: str, query: str, k: int = 3) -> list[dict]:
    """BM25 search inside one paper's PDF for passages matching `query`.

    Use this when the wiki page summary doesn't have the specific detail
    the user asked for — e.g. an exact number, a method parameter, a
    paragraph the page didn't quote. The chunk index is built lazily on
    first call and cached under `.grade-cache/{stem}/`.

    Each hit: {chunk_id, score, text, page_start, page_end, section,
    provenance}. `provenance` is the display form ('§results, p. 7') and is
    empty for a PDF whose headings/pages couldn't be resolved. On failure
    returns a single-element list whose only entry has a `note` field
    explaining why.

    Example: pdf_section_search("smith-2024-...", "training data composition", k=3)
    """
    try:
        build_pdf_index(stem)
    except FileNotFoundError:
        return [{"note": f"no PDF at papers/{stem}.pdf"}]
    except Exception as e:
        return [{"note": f"index build failed: {e}"}]
    try:
        chunks = query_pdf(stem, query, topk=k)
    except Exception as e:
        return [{"note": f"query failed: {e}"}]
    return [
        {
            "chunk_id": c.chunk_id,
            "score": float(c.score),
            "text": c.text,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section": c.section,
            "provenance": c.provenance(),
        }
        for c in chunks
    ]


# ---------- claim_lookup ----------

_STOPWORDS = {
    "a", "an", "the", "of", "for", "with", "and", "or", "in", "on", "at",
    "to", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "their", "our", "we",
    "they", "he", "she", "his", "her", "them", "than", "but", "not", "no",
    "do", "does", "did", "have", "has", "had", "can", "could", "would",
    "should", "may", "might", "will", "what", "which", "who", "how", "why",
    "when", "where",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def _tokenize_query(q: str) -> list[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(q)]
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def claims_by_stem(stem: str, *, include_context: bool = False) -> list[dict]:
    """Return every graded claim for one paper, sorted by section then position.

    Use when authoring a synthesis page that references this paper — the
    claims here are the paper's pre-graded, citable units. Cite each with
    `[[stem#claim_slug]]` (content-addressed; survives `db rebuild`).
    Cross-ref claims (skipped by the grader) are excluded so the output is
    the cite-ready set.

    Each hit has the same shape as `claim_lookup` minus `match_score`. The
    `bm25_top1` / `semantic_score` columns are NULL when grading hasn't run
    on this paper yet — surface but don't filter on them, so the caller can
    detect un-graded papers.

    `include_context=True` adds `supporting_text` (the verbatim chunk) and
    `supporting_provenance` (where in the PDF it sits) — the verbatim
    source-PDF chunk the grader matched to. Useful when an LLM judge needs
    the experimental setting alongside the bare claim (off by default to keep
    JSON dumps narrow).
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, paper_stem, section, position, text, claim_slug,
                   semantic_score, bm25_top1, bm25_top1_chunk_id,
                   supporting_text, supporting_provenance, last_graded_at
              FROM claims
             WHERE paper_stem = ? AND is_cross_ref = 0
             ORDER BY
                 CASE section
                     WHEN 'key_contributions' THEN 0
                     WHEN 'results' THEN 1
                     ELSE 2
                 END,
                 position
            """,
            (stem,),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        item = {
            "claim_slug": r["claim_slug"],
            "paper_stem": r["paper_stem"],
            "section": r["section"],
            "position": r["position"],
            "text": r["text"],
            "semantic_score": r["semantic_score"],
            "bm25_top1": r["bm25_top1"],
            "pdf_anchor_chunk_id": r["bm25_top1_chunk_id"],
            "graded": r["last_graded_at"] is not None,
        }
        if include_context:
            item["supporting_text"] = r["supporting_text"]
            item["supporting_provenance"] = r["supporting_provenance"]
        out.append(item)
    return out


def claim_lookup(query: str, k: int = 5, *, include_context: bool = False) -> list[dict]:
    """Find atomic claims (Key Contribution / Results bullets) matching the query.

    This is the grounded-citation primitive. Each claim is anchored to a
    specific paper + section + position, has a content-addressed
    `claim_slug`, and carries a graded score (semantic + BM25) showing how
    well the claim is supported by its source PDF. Cite hits as
    `[[stem#claim_slug]]` — durable, survives `db rebuild`.

    Primary backend is the `claims_fts` FTS5 index (`bm25()` ranking, OR-
    match across tokens). Falls back to a Python-side LIKE token counter
    when FTS5 is unavailable (stripped SQLite build) — same shape of hit
    dict, ordering degrades to hit-count.

    Each hit: {claim_slug, paper_stem, section, position, text, semantic_score,
    bm25_top1, pdf_anchor_chunk_id, match_score}.
    `match_score` is BM25 (FTS) or hit count (LIKE fallback) — always ≥ 1
    for returned rows. Falls back to ordering by `semantic_score` when
    scores tie.

    `include_context=True` adds `supporting_text` (the verbatim chunk) and
    `supporting_provenance` (where in the PDF it sits) — the verbatim
    source-PDF chunk the grader matched to (≤500 chars). Off by default.

    Returns [] if no claims match. Use this BEFORE `pdf_section_search` —
    claims are pre-graded and pre-vetted.

    Example: claim_lookup("AlphaFold 3 accuracy on protein-ligand", k=5)
    """
    tokens = _tokenize_query(query)
    if not tokens:
        return []
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        rows = _claim_lookup_fts(conn, tokens, k)
        if rows is None:
            rows = _claim_lookup_like(conn, tokens, k)
    finally:
        conn.close()
    out = []
    for r in rows:
        item = {
            "claim_slug": r["claim_slug"],
            "paper_stem": r["paper_stem"],
            "section": r["section"],
            "position": r["position"],
            "text": r["text"],
            "semantic_score": r["semantic_score"],
            "bm25_top1": r["bm25_top1"],
            "pdf_anchor_chunk_id": r["bm25_top1_chunk_id"],
            "match_score": r["match_score"],
        }
        if include_context:
            item["supporting_text"] = r["supporting_text"]
            item["supporting_provenance"] = r["supporting_provenance"]
        out.append(item)
    return out


def _claim_lookup_fts(conn, tokens: list[str], k: int):
    """FTS5 path. Returns rows on success, None if the FTS table is missing.

    Query form: `"tok1" OR "tok2" OR …` — quoting each token defuses FTS5
    operators the user may have typed (`AND`, `OR`, `NEAR`, `*`, `-`)
    while preserving OR-match semantics equivalent to the LIKE fallback.
    Ranking is BM25 (negative — smaller is better in FTS5; we negate so
    match_score is positive and higher = better).
    """
    fts_query = " OR ".join(f'"{t}"' for t in tokens)
    sql = """
        SELECT c.id, c.paper_stem, c.section, c.position, c.text, c.claim_slug,
               c.semantic_score, c.bm25_top1, c.bm25_top1_chunk_id,
               c.supporting_text, c.supporting_provenance,
               -bm25(claims_fts) AS match_score
        FROM claims_fts
        JOIN claims c ON c.id = claims_fts.rowid
        WHERE claims_fts MATCH ?
          AND c.is_cross_ref = 0
        ORDER BY match_score DESC,
                 COALESCE(c.semantic_score, 0) DESC,
                 COALESCE(c.bm25_top1, 0) DESC
        LIMIT ?
    """
    try:
        return conn.execute(sql, (fts_query, k)).fetchall()
    except sqlite3.OperationalError as e:
        # Signal fallback ONLY on FTS5-specific errors. Narrowly matches
        # "no such module: fts5" (FTS5 not compiled in) and any error
        # mentioning `claims_fts` (missing / corrupt virtual table). A
        # broader `"no such table" in msg` would swallow errors about the
        # underlying `claims` table too, promoting a real bug into a
        # confusing double-attempt failure via the LIKE path.
        msg = str(e).lower()
        if "claims_fts" in msg or ("no such module" in msg and "fts5" in msg):
            return None
        raise


def _claim_lookup_like(conn, tokens: list[str], k: int):
    """LIKE fallback — the original behavior when FTS5 isn't available."""
    like_clauses = " + ".join(
        ["(CASE WHEN LOWER(text) LIKE ? THEN 1 ELSE 0 END)"] * len(tokens)
    )
    params = [f"%{t}%" for t in tokens]
    sql = f"""
        SELECT id, paper_stem, section, position, text, claim_slug,
               semantic_score, bm25_top1, bm25_top1_chunk_id,
               supporting_text, supporting_provenance,
               ({like_clauses}) AS match_score
        FROM claims
        WHERE is_cross_ref = 0 AND ({like_clauses}) > 0
        ORDER BY match_score DESC,
                 COALESCE(semantic_score, 0) DESC,
                 COALESCE(bm25_top1, 0) DESC
        LIMIT ?
    """
    return conn.execute(sql, params + params + [k]).fetchall()
