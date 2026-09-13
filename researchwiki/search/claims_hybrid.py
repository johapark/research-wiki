"""Semantic and hybrid retrieval over grounded claims.

The claims table remains authoritative for text and durable anchors. This module
adds query-to-claim cosine similarity and RRF fusion with the existing FTS
lookup, reusing the framework's claim embedding cache.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from ..db import get_connection
from ..index import claim_embeddings
from ..index.embeddings import embed_texts
from .tools import claim_lookup


RRF_K = 60
PER_RANKER_DEPTH = 50


def _rows(*, include_context: bool) -> list[dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        db_rows = conn.execute(
            """
            SELECT paper_stem, section, position, text, claim_slug,
                   semantic_score, bm25_top1, bm25_top1_chunk_id,
                   supporting_text, supporting_provenance, last_graded_at
            FROM claims
            WHERE is_cross_ref = 0
            ORDER BY paper_stem, section, position
            """
        ).fetchall()
    finally:
        conn.close()
    out: list[dict] = []
    for row in db_rows:
        item = {
            "claim_slug": row["claim_slug"],
            "paper_stem": row["paper_stem"],
            "section": row["section"],
            "position": row["position"],
            "text": row["text"],
            # Claim-to-PDF support; query similarity is stored separately.
            "semantic_score": row["semantic_score"],
            "bm25_top1": row["bm25_top1"],
            "pdf_anchor_chunk_id": row["bm25_top1_chunk_id"],
            "graded": row["last_graded_at"] is not None,
        }
        if include_context:
            item["supporting_text"] = row["supporting_text"]
            item["supporting_provenance"] = row["supporting_provenance"]
        out.append(item)
    return out


def semantic_claim_lookup(
    query: str, k: int = 5, *, include_context: bool = False,
) -> list[dict]:
    """Return claims ranked by query-to-claim cosine similarity."""
    if not query.strip() or k <= 0:
        return []
    rows = _rows(include_context=include_context)
    if not rows:
        return []
    vectors = claim_embeddings.get_claim_embeddings(rows)
    query_vector = embed_texts([query])
    if vectors is None or query_vector is None or query_vector.size == 0:
        return []
    if vectors.shape[1] != query_vector.shape[1]:
        return []
    similarities = vectors @ query_vector[0]
    out: list[dict] = []
    for idx in np.argsort(-similarities)[:k]:
        item = dict(rows[int(idx)])
        item["query_semantic_score"] = float(
            max(0.0, min(1.0, similarities[int(idx)]))
        )
        item["match_score"] = item["query_semantic_score"]
        out.append(item)
    return out


def hybrid_claim_lookup(
    query: str, k: int = 5, *, include_context: bool = False,
) -> list[dict]:
    """Fuse lexical and semantic claim rankings with reciprocal rank fusion."""
    if k <= 0:
        return []
    depth = max(k, PER_RANKER_DEPTH)
    lexical = claim_lookup(query, k=depth, include_context=include_context)
    semantic = semantic_claim_lookup(query, k=depth, include_context=include_context)
    if not semantic:
        return lexical[:k]
    if not lexical:
        return semantic[:k]

    def identity(hit: dict) -> tuple[str, str, int]:
        return hit["paper_stem"], hit["section"], int(hit["position"])

    scores: dict[tuple[str, str, int], float] = {}
    items: dict[tuple[str, str, int], dict] = {}
    for rank, hit in enumerate(lexical, 1):
        ident = identity(hit)
        scores[ident] = scores.get(ident, 0.0) + 1.0 / (RRF_K + rank)
        items[ident] = dict(hit)
        items[ident]["lexical_rank"] = rank
    for rank, hit in enumerate(semantic, 1):
        ident = identity(hit)
        scores[ident] = scores.get(ident, 0.0) + 1.0 / (RRF_K + rank)
        slot = items.setdefault(ident, dict(hit))
        slot["semantic_rank"] = rank
        slot["query_semantic_score"] = hit["query_semantic_score"]

    out: list[dict] = []
    for ident in sorted(scores, key=lambda value: -scores[value])[:k]:
        item = items[ident]
        item["rrf_score"] = scores[ident]
        item["match_score"] = scores[ident]
        item.setdefault("lexical_rank", None)
        item.setdefault("semantic_rank", None)
        out.append(item)
    return out


def claim_query(
    query: str,
    k: int = 5,
    *,
    mode: str = "bm25",
    include_context: bool = False,
) -> list[dict]:
    """Unified claim retriever with graceful semantic-to-lexical fallback."""
    if mode == "bm25":
        return claim_lookup(query, k=k, include_context=include_context)
    if mode == "semantic":
        hits = semantic_claim_lookup(query, k=k, include_context=include_context)
        return hits or claim_lookup(query, k=k, include_context=include_context)
    if mode == "hybrid":
        return hybrid_claim_lookup(query, k=k, include_context=include_context)
    raise ValueError(f"unknown claim retrieval mode: {mode}")
