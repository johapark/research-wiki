"""Retrieval-fixture scorer + embedding bridge.

Complementary to `scorer.py`. Where `scorer.py` evaluates page-content
coverage (did the ingest pipeline extract the right things from the
PDF?), this module evaluates retrieval quality (does the embedding put
the right items in the top-K for a given query?).

Two fixture flavors, both loaded by `fixture.load_fixture` and dispatched
on `fixture_type`:
  - `claims` — anchored on (paper_stem, section, position); scored
               against the framework's claims-table retrieval.
  - `pages`  — anchored on paper_stem; scored against page-level search.

Designed to be embedding-agnostic: scoring takes already-retrieved top-K
lists, and the embedding bridge (`retrieve_claims` / `retrieve_pages`)
takes a model name + retrieval backend so the same fixtures evaluate any
sentence-transformers model.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..db import get_connection
from ..paths import wiki_root
from .fixture import Importance, RetrievalFixture
from ..log import log


SemanticBackend = Literal["bm25", "semantic", "hybrid"]
_GAIN_FOR_TIER: dict[Importance, int] = {"critical": 3, "high": 2, "normal": 1}


# ---------------------------------------------------------------------------
# Retrieved item models — what a retriever returns to the scorer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedClaim:
    paper_stem: str
    section: str
    position: int
    score: float


@dataclass(frozen=True)
class RetrievedPage:
    paper_stem: str   # may include category prefix
    score: float


# ---------------------------------------------------------------------------
# Score model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemResult:
    """Per-expected-anchor outcome — drives the diagnostic per-fixture report."""
    expected_key: str            # human-readable: "stem§section#pos" or "stem"
    importance: Importance
    rank: int | None             # 1-indexed; None if missing from top-K
    rationale: str
    rank_violation: bool = False  # set when expected_rank was specified and not met


@dataclass
class RetrievalScore:
    fixture_id: str
    fixture_type: str
    embedding: str
    backend: str
    k: int

    # Aggregate metrics
    mrr: float                          # reciprocal rank of TOP expected hit
    ndcg_at_k: float                    # importance-weighted nDCG@K
    expected_recall: float              # |expected ∩ top-K| / |expected|
    expected_recall_critical: float     # critical-only recall

    # Failure-mode counts
    must_not_hits: int                  # negative anchors that landed in top-K
    rank_violations: int                # items with expected_rank set but landed elsewhere

    # Per-item detail (for diagnosis)
    per_item: list[ItemResult]

    # Top-K returned (so JSON output can show what landed where)
    retrieved_keys: list[str] = field(default_factory=list)

    # Stems from the fixture's `must_not_appear` list — the diff uses this
    # to identify which negative anchors entered/left top-K, distinct from
    # neutral-but-unexpected items that just happened to rank.
    negative_stems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Scoring — pure logic, no embedding deps
# ---------------------------------------------------------------------------

def _format_claim_key(stem: str, section: str, position: int) -> str:
    return f"{stem}§{section}#{position}"


def _ndcg_at_k(gains_at_rank: list[int], ideal_gains: list[int], k: int) -> float:
    """nDCG@K with binary `relevant`/`not relevant` gains weighted by tier.

    DCG = sum_{i=1..k} (gain_i / log2(i+1))
    iDCG = same but on the ideal ordering of expected gains
    nDCG = DCG / iDCG (or 0 if iDCG == 0)
    """
    def _dcg(gains: list[int]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))

    actual_dcg = _dcg(gains_at_rank)
    ideal = sorted(ideal_gains, reverse=True)
    ideal_dcg = _dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def score_claims_fixture(
    fixture: RetrievalFixture,
    retrieved: list[RetrievedClaim],
    embedding_label: str,
    backend: SemanticBackend = "semantic",
) -> RetrievalScore:
    """Score a claims-fixture against an actual top-K result list."""
    if fixture.fixture_type != "claims":
        raise ValueError(f"score_claims_fixture: expected claims fixture, got {fixture.fixture_type!r}")

    k = fixture.k
    top_k = retrieved[:k]

    # rank map keyed on (stem, section, position) — 1-indexed.
    key_to_rank: dict[tuple[str, str, int], int] = {}
    for i, r in enumerate(top_k):
        key_to_rank.setdefault((r.paper_stem, r.section, r.position), i + 1)

    # Per-expected-claim outcomes
    per_item: list[ItemResult] = []
    gains_at_rank = [0] * k                              # gain at each rank position
    ideal_gains = [_GAIN_FOR_TIER[c.importance] for c in fixture.expected_claims]
    n_critical = sum(1 for c in fixture.expected_claims if c.importance == "critical")
    hits_total = 0
    hits_critical = 0
    best_critical_rank: int | None = None
    best_any_rank: int | None = None

    for c in fixture.expected_claims:
        rank = key_to_rank.get(c.key())
        per_item.append(ItemResult(
            expected_key=_format_claim_key(c.paper_stem, c.section, c.position),
            importance=c.importance,
            rank=rank,
            rationale=c.rationale,
        ))
        if rank is not None:
            hits_total += 1
            if c.importance == "critical":
                hits_critical += 1
                if best_critical_rank is None or rank < best_critical_rank:
                    best_critical_rank = rank
            if best_any_rank is None or rank < best_any_rank:
                best_any_rank = rank
            gains_at_rank[rank - 1] += _GAIN_FOR_TIER[c.importance]

    # MRR: prefer critical hits over any hits.
    if best_critical_rank is not None:
        mrr = 1.0 / best_critical_rank
    elif best_any_rank is not None:
        mrr = 1.0 / best_any_rank
    else:
        mrr = 0.0

    ndcg = _ndcg_at_k(gains_at_rank, ideal_gains, k)
    recall_all = hits_total / max(1, len(fixture.expected_claims))
    recall_crit = hits_critical / max(1, n_critical) if n_critical else 1.0

    # must_not_appear: any retrieved claim whose stem matches a negative
    # anchor counts as a violation. Stem-level — we don't require the
    # specific section/position to match.
    negative_stems = {n.paper_stem for n in fixture.must_not_appear}
    must_not_hits = sum(1 for r in top_k if r.paper_stem in negative_stems)

    retrieved_keys = [
        _format_claim_key(r.paper_stem, r.section, r.position) for r in top_k
    ]

    return RetrievalScore(
        fixture_id=fixture.fixture_id,
        fixture_type="claims",
        embedding=embedding_label,
        backend=backend,
        k=k,
        mrr=mrr,
        ndcg_at_k=ndcg,
        expected_recall=recall_all,
        expected_recall_critical=recall_crit,
        must_not_hits=must_not_hits,
        rank_violations=0,  # claims fixtures don't use expected_rank
        per_item=per_item,
        retrieved_keys=retrieved_keys,
        negative_stems=[n.paper_stem for n in fixture.must_not_appear],
    )


def _bare_stem(s: str) -> str:
    """Strip a leading `category/` prefix. Page keys come from the index in
    `category/stem` form, but fixtures may write either `category/stem` or
    just `stem`. Matching falls back to bare-stem when the exact form misses."""
    return s.split("/", 1)[-1] if "/" in s else s


def score_pages_fixture(
    fixture: RetrievalFixture,
    retrieved: list[RetrievedPage],
    embedding_label: str,
    backend: SemanticBackend = "semantic",
) -> RetrievalScore:
    """Score a pages-fixture. Same shape as score_claims_fixture but keyed
    on paper_stem only. Honors `expected_rank` if set."""
    if fixture.fixture_type != "pages":
        raise ValueError(f"score_pages_fixture: expected pages fixture, got {fixture.fixture_type!r}")

    k = fixture.k
    top_k = retrieved[:k]

    # Maintain both exact-key and bare-stem maps so fixtures can write either form.
    stem_to_rank: dict[str, int] = {}
    bare_to_rank: dict[str, int] = {}
    for i, r in enumerate(top_k):
        stem_to_rank.setdefault(r.paper_stem, i + 1)
        bare_to_rank.setdefault(_bare_stem(r.paper_stem), i + 1)

    per_item: list[ItemResult] = []
    gains_at_rank = [0] * k
    ideal_gains = [_GAIN_FOR_TIER[p.importance] for p in fixture.expected_pages]
    n_critical = sum(1 for p in fixture.expected_pages if p.importance == "critical")
    hits_total = 0
    hits_critical = 0
    best_critical_rank: int | None = None
    best_any_rank: int | None = None
    rank_violations = 0

    for p in fixture.expected_pages:
        # Try exact match first, then bare-stem fallback (so fixtures can
        # write either `khattab-2020-...` or `ai/khattab-2020-...`).
        rank = stem_to_rank.get(p.paper_stem)
        if rank is None:
            rank = bare_to_rank.get(_bare_stem(p.paper_stem))
        violated = (
            p.expected_rank is not None
            and rank is not None
            and rank != p.expected_rank
        )
        if violated:
            rank_violations += 1
        per_item.append(ItemResult(
            expected_key=p.paper_stem,
            importance=p.importance,
            rank=rank,
            rationale=p.rationale,
            rank_violation=violated,
        ))
        if rank is not None:
            hits_total += 1
            if p.importance == "critical":
                hits_critical += 1
                if best_critical_rank is None or rank < best_critical_rank:
                    best_critical_rank = rank
            if best_any_rank is None or rank < best_any_rank:
                best_any_rank = rank
            gains_at_rank[rank - 1] += _GAIN_FOR_TIER[p.importance]

    if best_critical_rank is not None:
        mrr = 1.0 / best_critical_rank
    elif best_any_rank is not None:
        mrr = 1.0 / best_any_rank
    else:
        mrr = 0.0

    ndcg = _ndcg_at_k(gains_at_rank, ideal_gains, k)
    recall_all = hits_total / max(1, len(fixture.expected_pages))
    recall_crit = hits_critical / max(1, n_critical) if n_critical else 1.0

    negative_stems = {n.paper_stem for n in fixture.must_not_appear}
    negative_bare = {_bare_stem(n.paper_stem) for n in fixture.must_not_appear}
    must_not_hits = sum(
        1 for r in top_k
        if r.paper_stem in negative_stems or _bare_stem(r.paper_stem) in negative_bare
    )

    retrieved_keys = [r.paper_stem for r in top_k]

    return RetrievalScore(
        fixture_id=fixture.fixture_id,
        fixture_type="pages",
        embedding=embedding_label,
        backend=backend,
        k=k,
        mrr=mrr,
        ndcg_at_k=ndcg,
        expected_recall=recall_all,
        expected_recall_critical=recall_crit,
        must_not_hits=must_not_hits,
        rank_violations=rank_violations,
        per_item=per_item,
        retrieved_keys=retrieved_keys,
        negative_stems=[n.paper_stem for n in fixture.must_not_appear],
    )


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------

@dataclass
class RetrievalDiff:
    """Pairwise comparison of two RetrievalScores on the same fixture."""
    fixture_id: str
    baseline_label: str
    candidate_label: str

    delta_mrr: float
    delta_ndcg: float
    delta_recall: float
    delta_recall_critical: float
    delta_must_not_hits: int

    # Per-item rank changes; (key, base_rank | None, cand_rank | None)
    improved: list[tuple[str, int | None, int | None]]
    regressed: list[tuple[str, int | None, int | None]]
    unchanged: list[str]

    # Negative-anchor changes (entries that left or entered top-K)
    must_not_left: list[str]    # in baseline top-K, not in candidate top-K (improvement)
    must_not_entered: list[str] # entered candidate top-K (regression)

    def to_dict(self) -> dict:
        return asdict(self)


def diff_retrieval_scores(
    baseline: RetrievalScore,
    candidate: RetrievalScore,
) -> RetrievalDiff:
    if baseline.fixture_id != candidate.fixture_id:
        raise ValueError(
            f"fixture_id mismatch: baseline={baseline.fixture_id} "
            f"candidate={candidate.fixture_id}"
        )

    base_ranks = {it.expected_key: it.rank for it in baseline.per_item}
    cand_ranks = {it.expected_key: it.rank for it in candidate.per_item}

    improved: list[tuple[str, int | None, int | None]] = []
    regressed: list[tuple[str, int | None, int | None]] = []
    unchanged: list[str] = []

    for key in base_ranks:
        b, c = base_ranks[key], cand_ranks.get(key)
        if b == c:
            unchanged.append(key)
        else:
            # Treat None (missing) as worse than any rank.
            base_score = -math.inf if b is None else -b
            cand_score = -math.inf if c is None else -c
            if cand_score > base_score:
                improved.append((key, b, c))
            else:
                regressed.append((key, b, c))

    # Negative-anchor membership change. Use stem-only matching (claim keys
    # are 'stem§sec#pos'; page keys are 'category/stem' or 'stem'); the
    # fixture's `must_not_appear` is stem-only so we normalize both sides.
    base_neg_stems = {_bare_stem(s) for s in baseline.negative_stems}
    cand_neg_stems = {_bare_stem(s) for s in candidate.negative_stems}
    base_neg_in_topk = {_stem_only(k) for k in baseline.retrieved_keys
                        if _bare_stem(_stem_only(k)) in base_neg_stems}
    cand_neg_in_topk = {_stem_only(k) for k in candidate.retrieved_keys
                        if _bare_stem(_stem_only(k)) in cand_neg_stems}
    must_not_left = sorted(base_neg_in_topk - cand_neg_in_topk)
    must_not_entered = sorted(cand_neg_in_topk - base_neg_in_topk)

    return RetrievalDiff(
        fixture_id=baseline.fixture_id,
        baseline_label=baseline.embedding,
        candidate_label=candidate.embedding,
        delta_mrr=candidate.mrr - baseline.mrr,
        delta_ndcg=candidate.ndcg_at_k - baseline.ndcg_at_k,
        delta_recall=candidate.expected_recall - baseline.expected_recall,
        delta_recall_critical=candidate.expected_recall_critical - baseline.expected_recall_critical,
        delta_must_not_hits=candidate.must_not_hits - baseline.must_not_hits,
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
        must_not_left=must_not_left,
        must_not_entered=must_not_entered,
    )


def _stem_only(key: str) -> str:
    """Strip section/position from a key (claim keys are 'stem§sec#pos')."""
    return key.split("§", 1)[0]


# ---------------------------------------------------------------------------
# Aggregation across fixture sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixtureSetSummary:
    embedding: str
    backend: str
    n_fixtures: int
    mean_mrr: float
    mean_ndcg: float
    mean_recall: float
    mean_recall_critical: float
    total_must_not_hits: int
    per_fixture: list[RetrievalScore]


def summarize(scores: list[RetrievalScore], embedding_label: str, backend: str) -> FixtureSetSummary:
    if not scores:
        return FixtureSetSummary(
            embedding=embedding_label, backend=backend, n_fixtures=0,
            mean_mrr=0.0, mean_ndcg=0.0, mean_recall=0.0, mean_recall_critical=0.0,
            total_must_not_hits=0, per_fixture=[],
        )
    n = len(scores)
    return FixtureSetSummary(
        embedding=embedding_label,
        backend=backend,
        n_fixtures=n,
        mean_mrr=sum(s.mrr for s in scores) / n,
        mean_ndcg=sum(s.ndcg_at_k for s in scores) / n,
        mean_recall=sum(s.expected_recall for s in scores) / n,
        mean_recall_critical=sum(s.expected_recall_critical for s in scores) / n,
        total_must_not_hits=sum(s.must_not_hits for s in scores),
        per_fixture=list(scores),
    )


# ---------------------------------------------------------------------------
# Embedding bridge
# ---------------------------------------------------------------------------
#
# Two parallel concerns: (1) loading sentence-transformers models by name,
# bypassing the framework's default-only singleton; (2) a per-model on-disk
# cache so repeated A/B runs don't re-embed.

_MODEL_CACHE: dict[str, object] = {}


def _model_slug(model_name: str) -> str:
    """Disk-safe slug for a model id — strip the org prefix.
    `BAAI/bge-small-en-v1.5` → `bge-small-en-v1.5`."""
    return model_name.split("/", 1)[-1]


def cache_dir_for(model_name: str) -> Path:
    """Per-model retrieval cache directory under the wiki root.

    Sharing the framework default `.semantic-cache/` across models would
    silently overwrite vectors when the model changes. Per-model dirs
    let A/B runs use both caches concurrently."""
    return wiki_root() / f".retrieval-cache-{_model_slug(model_name)}"


def _load_model(model_name: str, *, trust_remote_code: bool = False):
    """Lazy-load a sentence-transformers model by name. Per-name singleton
    so repeat calls reuse — `trust_remote_code` only affects the FIRST load
    of a given model (once a model is in cache, subsequent calls reuse it).
    Returns None on failure.

    `trust_remote_code=True` is required by some models (Nomic family) that
    ship custom modeling code in the HF repo. Off by default — opt in only
    for trusted model providers."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:
        log(f"sentence-transformers not installed: {e}", tag="retrieval")
        return None
    try:
        model = SentenceTransformer(
            model_name, device="cpu", trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        log(f"failed to load model {model_name!r}: {e}", tag="retrieval")
        return None
    _MODEL_CACHE[model_name] = model
    return model


def _embed_with(
    texts: list[str],
    model_name: str,
    *,
    trust_remote_code: bool = False,
    prefix: str = "",
) -> np.ndarray | None:
    """Embed `texts` with the named model. If `prefix` is non-empty, prepend
    it to every text — used for instruction-tuned embeddings (Nomic, E5,
    Instructor, etc.) that distinguish indexing vs querying via prefixes."""
    model = _load_model(model_name, trust_remote_code=trust_remote_code)
    if model is None:
        return None
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if prefix:
        texts = [prefix + t for t in texts]
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embs.astype(np.float32, copy=False)


# Per-model claim cache layout under cache_dir_for(model):
#   claims.npy        float32, shape (N, dim), L2-normalized
#   claims_meta.json  {model, dim, built_at, rows: [{paper_stem, section, position, text}, ...]}

def _claim_cache_paths(model_name: str) -> tuple[Path, Path]:
    d = cache_dir_for(model_name)
    return d / "claims.npy", d / "claims_meta.json"


def _all_db_claims() -> list[dict]:
    """Pull every non-cross-ref claim from the state DB."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT paper_stem, section, position, text FROM claims "
            "WHERE is_cross_ref = 0 ORDER BY paper_stem, section, position"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _load_or_build_claim_cache(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    doc_prefix: str = "",
) -> tuple[np.ndarray, list[dict]] | None:
    """Load claims embeddings from disk or build them. The cache meta records
    `doc_prefix`; if the requested prefix differs from what's on disk, rebuild
    — embeddings must match the prefix used at index time."""
    npy, meta = _claim_cache_paths(model_name)
    if npy.exists() and meta.exists():
        try:
            meta_d = json.loads(meta.read_text())
            cached_prefix = meta_d.get("doc_prefix", "")
            arr = np.load(npy)
            rows = meta_d.get("rows", [])
            if (
                arr.shape[0] == len(rows)
                and len(rows) > 0
                and cached_prefix == doc_prefix
            ):
                return arr, rows
            if cached_prefix != doc_prefix:
                log(
                    f"doc_prefix changed for {model_name} "
                    f"({cached_prefix!r} → {doc_prefix!r}); rebuilding cache", tag="retrieval"
                )
        except Exception:
            pass  # rebuild
    rows = _all_db_claims()
    if not rows:
        log("claims table is empty — run `researchwiki db rebuild` first", tag="retrieval")
        return None
    texts = [r["text"] for r in rows]
    arr = _embed_with(
        texts, model_name,
        trust_remote_code=trust_remote_code, prefix=doc_prefix,
    )
    if arr is None or arr.shape[0] != len(rows):
        return None
    cache_dir_for(model_name).mkdir(parents=True, exist_ok=True)
    np.save(npy, arr)
    meta.write_text(json.dumps({
        "model": model_name,
        "dim": int(arr.shape[1]),
        "built_at": int(time.time()),
        "doc_prefix": doc_prefix,
        "rows": rows,
    }, indent=2))
    return arr, rows


# Per-model page cache layout — mirrors the framework's `.semantic-cache/`
# but parameterized by model. Same fields as `pages_meta.json` in
# index.pages_semantic so we could share the loader if needed later.

def _page_cache_paths(model_name: str) -> tuple[Path, Path]:
    d = cache_dir_for(model_name)
    return d / "pages.npy", d / "pages_meta.json"


def _load_or_build_page_cache(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    doc_prefix: str = "",
) -> tuple[np.ndarray, list[dict]] | None:
    npy, meta = _page_cache_paths(model_name)
    if npy.exists() and meta.exists():
        try:
            meta_d = json.loads(meta.read_text())
            cached_prefix = meta_d.get("doc_prefix", "")
            arr = np.load(npy)
            rows = meta_d.get("rows", [])
            if (
                arr.shape[0] == len(rows)
                and len(rows) > 0
                and cached_prefix == doc_prefix
            ):
                return arr, rows
            if cached_prefix != doc_prefix:
                log(
                    f"doc_prefix changed for {model_name} "
                    f"({cached_prefix!r} → {doc_prefix!r}); rebuilding cache", tag="retrieval"
                )
        except Exception:
            pass  # rebuild

    # Reuse the framework's page-text builder so the candidate cache
    # embeds the same fields the default semantic index does.
    from ..index.pages_semantic import page_index_text
    from ..wiki import read_pages

    pages = read_pages()
    rows: list[dict] = []
    texts: list[str] = []
    for p in pages:
        text = page_index_text(p)
        if not text.strip():
            continue
        rows.append({
            "key": p.key,
            "stem": p.stem,
            "category": p.category,
            "page_type": p.page_type,
            "title": p.fm.get("title", ""),
        })
        texts.append(text)

    if not rows:
        log("no indexable wiki pages", tag="retrieval")
        return None
    arr = _embed_with(
        texts, model_name,
        trust_remote_code=trust_remote_code, prefix=doc_prefix,
    )
    if arr is None or arr.shape[0] != len(rows):
        return None
    cache_dir_for(model_name).mkdir(parents=True, exist_ok=True)
    np.save(npy, arr)
    meta.write_text(json.dumps({
        "model": model_name,
        "dim": int(arr.shape[1]),
        "built_at": int(time.time()),
        "doc_prefix": doc_prefix,
        "rows": rows,
    }, indent=2))
    return arr, rows


# ---------------------------------------------------------------------------
# BM25-side: keyword retrieval over claim text + page text
# ---------------------------------------------------------------------------
#
# Used directly when `backend == "bm25"`, and as one half of `hybrid`. Light
# token-LIKE match over claim text (mirrors `tools.core.claim_lookup`'s
# scoring shape) for claims; for pages we delegate to the existing Tantivy
# index when available.

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "with", "and", "or", "in", "on", "at",
    "to", "from", "by", "as", "is", "are", "was", "were", "be", "do", "does",
})


def _tokenize_query(q: str) -> list[str]:
    out = []
    for tok in _TOKEN_RE.findall(q.lower()):
        if len(tok) >= 3 and tok not in _STOPWORDS:
            out.append(tok)
    return out


def _bm25ish_claims(query: str, k: int) -> list[RetrievedClaim]:
    """Cheap keyword retrieval over claim text. Match-count + claim's own
    semantic_score-against-PDF as a secondary signal. Returns a list of
    RetrievedClaim (not full claim_lookup output) for scorer consumption."""
    tokens = _tokenize_query(query)
    if not tokens:
        return []
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        like_clauses = " + ".join(
            ["(CASE WHEN LOWER(text) LIKE ? THEN 1 ELSE 0 END)"] * len(tokens)
        )
        params = [f"%{t}%" for t in tokens]
        sql = f"""
            SELECT paper_stem, section, position,
                   ({like_clauses}) AS match_score,
                   COALESCE(semantic_score, 0) AS sem
            FROM claims
            WHERE is_cross_ref = 0 AND ({like_clauses}) > 0
            ORDER BY match_score DESC, sem DESC
            LIMIT ?
        """
        rows = conn.execute(sql, params + params + [k]).fetchall()
    finally:
        conn.close()
    return [
        RetrievedClaim(
            paper_stem=r["paper_stem"], section=r["section"],
            position=r["position"],
            score=float(r["match_score"] + 0.01 * r["sem"]),  # primary by match, tie-break by semantic
        )
        for r in rows
    ]


def _bm25_pages(query: str, k: int) -> list[RetrievedPage]:
    """Keyword retrieval over wiki pages via the existing Tantivy index."""
    try:
        from ..index.pages_bm25 import TantivySearchBackend
    except Exception as e:
        log(f"BM25 page index unavailable: {e}", tag="retrieval")
        return []
    try:
        backend = TantivySearchBackend()
        hits = backend.query(query, limit=k)
    except Exception as e:
        log(f"tantivy query failed: {e}", tag="retrieval")
        return []
    return [RetrievedPage(paper_stem=h.key, score=float(h.score)) for h in hits]


# ---------------------------------------------------------------------------
# Public retrievers
# ---------------------------------------------------------------------------

def retrieve_claims(
    fixture: RetrievalFixture,
    model_name: str,
    backend: SemanticBackend = "semantic",
    *,
    trust_remote_code: bool = False,
    doc_prefix: str = "",
    query_prefix: str = "",
) -> list[RetrievedClaim]:
    """Run a claims-fixture's query against the candidate embedding.

    `backend`:
      - "semantic": pure cosine over candidate model's claim embeddings
      - "bm25":     keyword match over claim text (model-independent)
      - "hybrid":   RRF fusion of semantic + bm25

    `trust_remote_code`: opt in for models that ship custom HF code.
    `doc_prefix` / `query_prefix`: instruction prefixes for models that
    distinguish indexing vs querying (Nomic, E5, Instructor). Cache is
    invalidated when `doc_prefix` changes.
    """
    if backend == "bm25":
        return _bm25ish_claims(fixture.query, fixture.k)

    cache = _load_or_build_claim_cache(
        model_name, trust_remote_code=trust_remote_code, doc_prefix=doc_prefix,
    )
    if cache is None:
        return []
    arr, rows = cache

    q_emb = _embed_with(
        [fixture.query], model_name,
        trust_remote_code=trust_remote_code, prefix=query_prefix,
    )
    if q_emb is None or q_emb.shape[0] == 0 or q_emb.shape[1] != arr.shape[1]:
        if q_emb is not None and q_emb.shape[1] != arr.shape[1]:
            log(
                f"dim mismatch (query {q_emb.shape[1]} vs cache {arr.shape[1]}); "
                f"rebuilding cache may be needed", tag="retrieval"
            )
        return []
    sims = arr @ q_emb[0]                      # (N,)
    order = np.argsort(-sims)

    semantic_hits = [
        RetrievedClaim(
            paper_stem=rows[int(i)]["paper_stem"],
            section=rows[int(i)]["section"],
            position=int(rows[int(i)]["position"]),
            score=float(max(0.0, min(1.0, sims[int(i)]))),
        )
        for i in order[: max(fixture.k, 50)]   # over-fetch for hybrid fusion
    ]

    if backend == "semantic":
        return semantic_hits[: fixture.k]

    # hybrid: reciprocal-rank fusion with bm25 (k=60 RRF constant)
    bm25_hits = _bm25ish_claims(fixture.query, max(fixture.k, 50))
    return _rrf_claims(semantic_hits, bm25_hits, fixture.k)


def retrieve_pages(
    fixture: RetrievalFixture,
    model_name: str,
    backend: SemanticBackend = "semantic",
    *,
    trust_remote_code: bool = False,
    doc_prefix: str = "",
    query_prefix: str = "",
) -> list[RetrievedPage]:
    if backend == "bm25":
        return _bm25_pages(fixture.query, fixture.k)

    cache = _load_or_build_page_cache(
        model_name, trust_remote_code=trust_remote_code, doc_prefix=doc_prefix,
    )
    if cache is None:
        return []
    arr, rows = cache

    q_emb = _embed_with(
        [fixture.query], model_name,
        trust_remote_code=trust_remote_code, prefix=query_prefix,
    )
    if q_emb is None or q_emb.shape[0] == 0 or q_emb.shape[1] != arr.shape[1]:
        return []
    sims = arr @ q_emb[0]
    order = np.argsort(-sims)
    semantic_hits = [
        RetrievedPage(
            paper_stem=rows[int(i)]["key"],
            score=float(max(0.0, min(1.0, sims[int(i)]))),
        )
        for i in order[: max(fixture.k, 50)]
    ]
    if backend == "semantic":
        return semantic_hits[: fixture.k]

    bm25_hits = _bm25_pages(fixture.query, max(fixture.k, 50))
    return _rrf_pages(semantic_hits, bm25_hits, fixture.k)


# ---------------------------------------------------------------------------
# Reciprocal-rank fusion
# ---------------------------------------------------------------------------

_RRF_K = 60


def _rrf_claims(a: list[RetrievedClaim], b: list[RetrievedClaim], k: int) -> list[RetrievedClaim]:
    scores: dict[tuple[str, str, int], float] = {}
    items: dict[tuple[str, str, int], RetrievedClaim] = {}
    for rank, h in enumerate(a):
        key = (h.paper_stem, h.section, h.position)
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        items[key] = h
    for rank, h in enumerate(b):
        key = (h.paper_stem, h.section, h.position)
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        items.setdefault(key, h)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [
        RetrievedClaim(
            paper_stem=items[key].paper_stem,
            section=items[key].section,
            position=items[key].position,
            score=score,
        )
        for key, score in fused
    ]


def _rrf_pages(a: list[RetrievedPage], b: list[RetrievedPage], k: int) -> list[RetrievedPage]:
    scores: dict[str, float] = {}
    for rank, h in enumerate(a):
        scores[h.paper_stem] = scores.get(h.paper_stem, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, h in enumerate(b):
        scores[h.paper_stem] = scores.get(h.paper_stem, 0.0) + 1.0 / (_RRF_K + rank + 1)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [RetrievedPage(paper_stem=stem, score=score) for stem, score in fused]
