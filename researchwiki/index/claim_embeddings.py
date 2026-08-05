"""Cached bi-encoder embeddings for claims.

The claim-overlap cross-linker embeds every wiki claim to find near-duplicate
pairs on each ingest. Re-embedding the whole corpus (~1k claims) each run is
wasteful, so we cache one vector per claim keyed by its stable identity
(stem·section·position) plus a hash of the claim text — a text edit invalidates
just that row. Mirrors the page-level `pages.npy` store under `.semantic-cache/`.

The cache is a pure derived artifact: safe to delete (it rebuilds), never
committed. Deletions in the corpus are handled implicitly — each call rewrites
the cache to exactly the rows it was asked about, so stale claims fall out.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

from ..paths import semantic_cache_dir


def _paths() -> tuple[Path, Path]:
    d = semantic_cache_dir()
    return d / "claims.npy", d / "claims_meta.json"


def _identity(row: dict) -> str:
    return f"{row['paper_stem']}·{row['section']}·{row['position']}"


def _text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def load_cached_claim_embeddings(rows: list[dict]):
    """Cache-only sibling of `get_claim_embeddings` — never loads the bi-encoder.

    Returns `(vectors, row_indices)` where `vectors` is an
    (len(row_indices), dim) L2-normalized array and `row_indices` are the
    positions in `rows` it covers — i.e. exactly the claims already cached
    with a matching text hash. Returns None when numpy is missing or the
    cache is absent/unreadable/empty.

    Exists because `get_claim_embeddings` calls `embeddings.is_available()`,
    which constructs the SentenceTransformer (~3s) even when every row is a
    cache hit, and then rewrites the cache to its row set. Neither is
    acceptable on a read-only, always-on surface: `lint` must stay sub-second
    and must not mutate derived state. Callers here trade recall for that —
    a cold cache yields None and the caller skips its check rather than
    paying for a corpus embed.
    """
    if not _NUMPY or not rows:
        return None
    npy_path, meta_path = _paths()
    if not (npy_path.exists() and meta_path.exists()):
        return None
    try:
        vecs = np.load(npy_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    index = {m["id"]: (i, m["hash"]) for i, m in enumerate(meta)}

    picked_rows: list[int] = []
    picked_cache: list[int] = []
    for out_idx, row in enumerate(rows):
        hit = index.get(_identity(row))
        if hit is None:
            continue
        cache_idx, h = hit
        if h != _text_hash(row["text"]) or cache_idx >= len(vecs):
            continue
        picked_rows.append(out_idx)
        picked_cache.append(cache_idx)
    if not picked_rows:
        return None

    out = vecs[picked_cache].astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms, picked_rows


def get_claim_embeddings(rows: list[dict]):
    """Return an (len(rows), dim) L2-normalized array aligned to `rows`.

    `rows` are dicts with keys paper_stem / section / position / text. Reuses
    cached vectors for unchanged claims and embeds only new/edited ones, then
    rewrites the cache to the current row set. Returns None when numpy or the
    bi-encoder is unavailable (callers skip the overlap pass gracefully).
    """
    if not _NUMPY or not rows:
        return None
    from .embeddings import embed_texts, is_available
    if not is_available():
        return None

    npy_path, meta_path = _paths()

    # Load existing cache into an identity → (row_idx, hash) map.
    cached_vecs = None
    cached_index: dict[str, dict] = {}
    if npy_path.exists() and meta_path.exists():
        try:
            cached_vecs = np.load(npy_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for i, m in enumerate(meta):
                cached_index[m["id"]] = {"row": i, "hash": m["hash"]}
        except Exception:
            cached_vecs, cached_index = None, {}

    # Decide which rows need (re)embedding.
    reuse: dict[int, int] = {}          # out_idx -> cached_row_idx
    to_embed: list[int] = []            # out_idx needing a fresh vector
    for out_idx, row in enumerate(rows):
        ident, h = _identity(row), _text_hash(row["text"])
        hit = cached_index.get(ident)
        if hit is not None and hit["hash"] == h and cached_vecs is not None and hit["row"] < len(cached_vecs):
            reuse[out_idx] = hit["row"]
        else:
            to_embed.append(out_idx)

    fresh = None
    if to_embed:
        fresh = embed_texts([rows[i]["text"] for i in to_embed])
        if fresh is None or fresh.size == 0:
            return None

    dim = (fresh.shape[1] if fresh is not None
           else cached_vecs.shape[1] if cached_vecs is not None and len(cached_vecs) else None)
    if dim is None:
        return None

    out = np.zeros((len(rows), dim), dtype=np.float32)
    for out_idx, crow in reuse.items():
        out[out_idx] = cached_vecs[crow]
    for k, out_idx in enumerate(to_embed):
        out[out_idx] = fresh[k]

    _persist(rows, out, npy_path, meta_path)
    return out


def _persist(rows: list[dict], vecs, npy_path: Path, meta_path: Path) -> None:
    try:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, vecs)
        meta = [{"id": _identity(r), "hash": _text_hash(r["text"])} for r in rows]
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # cache is best-effort; a write failure just means we re-embed next time
