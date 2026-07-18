"""Bi-encoder semantic similarity scorer.

Replaces the prior MiniCheck NLI grader. Same role: a non-LLM grounding
signal orthogonal to the generative LLM, used to detect when the wiki page
diverges semantically from the source PDF. Different mechanism: a
sentence-transformers bi-encoder produces a fixed-size embedding for the
claim and for each PDF chunk, then we take the max cosine similarity over
the top-K retrieved chunks.

Why this swap:
  - ~133 MB instead of ~750 MB
  - CPU-only (~50 ms/sentence) — eliminates the MPS OOM that hit MiniCheck
    on long-evidence papers like xu-2025
  - chunk embeddings are cached at index-build time, so per-claim scoring
    is one query embedding + a NumPy dot product

What's lost vs NLI: cosine similarity doesn't naturally capture negation
or contradiction (cos("X works", "X doesn't work") is high). The
contradiction signal is recovered by `coverage._negation_mismatch`, a
deterministic check that runs alongside this scorer.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ..log import log


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DEVICE = "cpu"   # explicit — escaping MPS is the whole point


def _maybe_enable_offline_mode() -> None:
    """If `DEFAULT_MODEL` is already in the HF cache, set HF_HUB_OFFLINE=1.

    The HuggingFace Hub layer makes a revalidation network call on every
    SentenceTransformer construction, even when all weights are cached
    locally. That call (a) adds 1–3s per CLI invocation, (b) emits a
    noisy "unauthenticated requests" warning, and (c) breaks every
    `researchwiki` command if the user is offline.

    Setting `HF_HUB_OFFLINE=1` short-circuits the layer to local-only
    reads. We only set it when the model directory is already present —
    a first-ever run still gets to download.

    Respects an existing `HF_HUB_OFFLINE` value (don't override the user).
    """
    if "HF_HUB_OFFLINE" in os.environ:
        return
    hf_home = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    # HF cache layout: hub/models--{org}--{name}/{blobs,refs,snapshots}/...
    model_dir = hf_home / "hub" / f"models--{DEFAULT_MODEL.replace('/', '--')}"
    if model_dir.exists():
        os.environ["HF_HUB_OFFLINE"] = "1"


_maybe_enable_offline_mode()


@dataclass
class SemanticScore:
    score: float                # max cosine similarity over evidence chunks, in [0, 1]
    best_chunk_preview: str     # first 200 chars of the chunk that scored highest


_model_singleton = None


def _get_model(model_name: str = DEFAULT_MODEL):
    """Lazy-load the SentenceTransformer instance. Returns None if unavailable."""
    global _model_singleton
    if _model_singleton is not None:
        return _model_singleton
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:
        log(f"sentence-transformers not installed: {e}", tag="semantic")
        return None
    try:
        _model_singleton = SentenceTransformer(model_name, device=DEFAULT_DEVICE)
    except Exception as e:
        log(f"failed to load embedding model '{model_name}': {e}", tag="semantic")
        return None
    return _model_singleton


def is_available() -> bool:
    """Probe whether semantic scoring can run. Lazy-loads the model on first call."""
    return _get_model() is not None


def embed_texts(texts: list[str]) -> np.ndarray | None:
    """Embed a list of texts. Returns shape (n, dim) float32 with L2-normalized
    rows so cosine similarity reduces to a dot product. Returns None if the
    model can't be loaded.
    """
    model = _get_model()
    if model is None:
        return None
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    # normalize_embeddings=True returns unit-length vectors, so cosine == dot.
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embs.astype(np.float32, copy=False)


def score_claim(
    claim_text: str,
    evidence_chunks: list[str],
    *,
    chunk_embeddings: np.ndarray | None = None,
) -> SemanticScore | None:
    """Score one claim against retrieved evidence chunks.

    Args:
      claim_text: the bullet/sentence to verify.
      evidence_chunks: top-K chunk texts retrieved by BM25.
      chunk_embeddings: pre-computed normalized embeddings for `evidence_chunks`,
        shape (len(evidence_chunks), dim). When provided, we skip re-embedding
        the chunks — this is the fast path used by `coverage.py` once
        `pdf_index.get_chunk_embeddings` is wired up. When None, we embed the
        chunks here (slower; intended for ad-hoc / test calls).

    Returns SemanticScore with score = max cosine similarity over chunks. None
    if the model is unavailable or evidence is empty.
    """
    model = _get_model()
    if model is None:
        return None
    if not evidence_chunks:
        return SemanticScore(score=0.0, best_chunk_preview="")

    claim_emb = embed_texts([claim_text])
    if claim_emb is None or claim_emb.shape[0] == 0:
        return None

    if chunk_embeddings is None:
        chunk_embeddings = embed_texts(evidence_chunks)
        if chunk_embeddings is None:
            return None

    # Both arrays are L2-normalized, so cosine == dot product.
    sims = chunk_embeddings @ claim_emb[0]   # shape (n_chunks,)
    if sims.size == 0:
        return SemanticScore(score=0.0, best_chunk_preview="")
    best_idx = int(np.argmax(sims))
    # Clamp to [0, 1] — for normalized BGE embeddings sims are typically
    # already non-negative, but float noise can produce tiny negatives.
    best = float(max(0.0, min(1.0, sims[best_idx])))
    return SemanticScore(
        score=best,
        best_chunk_preview=evidence_chunks[best_idx][:200],
    )
