"""Hybrid retrieval = reciprocal rank fusion over BM25 + semantic page index.

Why RRF rather than score normalization: BM25 scores and cosine similarities
live on incompatible scales (~0–20 vs. 0–1) and their distributions vary by
query. RRF only uses *rank* per ranker, which is robust without per-query
calibration. The standard `k=60` constant is from Cormack et al. 2009.

Public API:
  hybrid_query(text, k)       — text → ranked HybridHits
  hybrid_more_like(key, k)    — seed page → ranked HybridHits
  is_hybrid_available()       — both backends ready?

Both fusion paths fall back gracefully:
  - If the semantic index is missing, we return BM25-only results.
  - If Tantivy is missing, we return semantic-only results.
  - If both are missing, the caller gets `[]` and should rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..index import pages_semantic as semantic_pages
from ..index.pages_bm25 import TantivySearchBackend
from ..index.types import SearchBackendUnavailable, SearchHit


# Standard RRF constant. Larger k spreads the fusion across deeper ranks (more
# tolerance for one ranker's idiosyncrasies); smaller k makes top-1 dominate.
RRF_K = 60

# Per-ranker depth: how many candidates to pull from each side before fusion.
# A larger pool catches relevant docs that one ranker scored poorly; deeper
# pools cost a bit more but BM25 + a 384-dim cosine over <1k pages is cheap.
PER_RANKER_DEPTH = 50


@dataclass
class HybridHit:
    """One fused result. `rrf_score` is the sum of 1/(k+rank) contributions
    from each ranker that returned this document; missing rankers contribute
    nothing. `bm25_rank` and `semantic_rank` are 1-indexed positions in
    their source rankings, or None when the doc didn't appear there."""
    key: str                   # category/stem
    stem: str
    category: str
    page_type: str
    title: str
    rrf_score: float
    bm25_rank: int | None
    bm25_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    snippet: str = ""

    def to_search_hit(self) -> SearchHit:
        """Lossy conversion for callers that want the prose `--like` formatter.
        Stores `rrf_score` as `score` so existing fmt code works."""
        return SearchHit(
            stem=self.stem,
            category=self.category,
            page_type=self.page_type,
            title=self.title,
            score=self.rrf_score,
            snippet=self.snippet,
        )


def is_hybrid_available() -> bool:
    """True iff both Tantivy and the semantic page index are built."""
    if not semantic_pages.index_exists():
        return False
    try:
        TantivySearchBackend().query("a", limit=1)   # sanity probe
    except SearchBackendUnavailable:
        return False
    return True


# ---------- fusion core ----------

def _fuse(
    bm25_hits: list[SearchHit],
    semantic_hits: list[semantic_pages.PageHit],
    k: int,
) -> list[HybridHit]:
    """Reciprocal rank fusion. Iterates each ranking, accumulates 1/(K+rank)
    contributions per key, then sorts descending."""
    by_key: dict[str, dict] = {}

    for rank, h in enumerate(bm25_hits, start=1):
        slot = by_key.setdefault(h.key, _empty_slot(h.key))
        slot["bm25_rank"] = rank
        slot["bm25_score"] = h.score
        slot["title"] = slot["title"] or h.title
        slot["snippet"] = slot["snippet"] or h.snippet
        slot["category"] = h.category
        slot["page_type"] = h.page_type
        slot["stem"] = h.stem
        slot["rrf_score"] += 1.0 / (RRF_K + rank)

    for rank, h in enumerate(semantic_hits, start=1):
        slot = by_key.setdefault(h.key, _empty_slot(h.key))
        slot["semantic_rank"] = rank
        slot["semantic_score"] = h.score
        slot["title"] = slot["title"] or h.title
        slot["category"] = h.category
        slot["page_type"] = h.page_type
        slot["stem"] = h.stem
        slot["rrf_score"] += 1.0 / (RRF_K + rank)

    fused = [
        HybridHit(
            key=key,
            stem=s["stem"] or key.split("/", 1)[-1],
            category=s["category"],
            page_type=s["page_type"],
            title=s["title"],
            rrf_score=s["rrf_score"],
            bm25_rank=s["bm25_rank"],
            bm25_score=s["bm25_score"],
            semantic_rank=s["semantic_rank"],
            semantic_score=s["semantic_score"],
            snippet=s["snippet"],
        )
        for key, s in by_key.items()
    ]
    fused.sort(key=lambda h: -h.rrf_score)
    return fused[:k]


def _empty_slot(key: str) -> dict:
    return {
        "stem": "",
        "category": key.split("/", 1)[0] if "/" in key else "",
        "page_type": "",
        "title": "",
        "snippet": "",
        "bm25_rank": None,
        "bm25_score": None,
        "semantic_rank": None,
        "semantic_score": None,
        "rrf_score": 0.0,
    }


# ---------- public entry points ----------

def hybrid_query(text: str, limit: int = 10) -> list[HybridHit]:
    """Free-form text → top-`limit` fused results.

    Pulls `PER_RANKER_DEPTH` from each side, fuses, returns top-`limit`.
    """
    bm25_hits: list[SearchHit] = []
    sem_hits: list[semantic_pages.PageHit] = []

    try:
        backend = TantivySearchBackend()
        bm25_hits = backend.query(text, limit=PER_RANKER_DEPTH)
    except SearchBackendUnavailable:
        bm25_hits = []

    if semantic_pages.index_exists():
        sem_hits = semantic_pages.query_text(text, k=PER_RANKER_DEPTH)

    return _fuse(bm25_hits, sem_hits, k=limit)


def hybrid_more_like(key: str, limit: int = 10) -> list[HybridHit]:
    """Seed page (category/stem) → top-`limit` fused See-Also.

    BM25 side uses Tantivy's `more_like` (title+summary as query). Semantic
    side uses the seed page's stored embedding directly via `query_stem` —
    no re-embedding, no quality loss from text round-trips.
    """
    bm25_hits: list[SearchHit] = []
    sem_hits: list[semantic_pages.PageHit] = []

    try:
        backend = TantivySearchBackend()
        bm25_hits = backend.more_like(key, limit=PER_RANKER_DEPTH)
    except SearchBackendUnavailable:
        bm25_hits = []

    if semantic_pages.index_exists():
        sem_hits = semantic_pages.query_stem(key, k=PER_RANKER_DEPTH)

    return _fuse(bm25_hits, sem_hits, k=limit)
