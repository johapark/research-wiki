"""Cached bi-encoder embeddings for claims.

The claim-overlap cross-linker embeds every wiki claim to find near-duplicate
pairs on each ingest. Re-embedding the whole corpus (~1k claims) each run is
wasteful, so we cache one vector per claim keyed by its stable identity
(stem·section·position) plus a hash of the claim text — a text edit invalidates
just that row. Uses one atomic `claims.npz` bundle under `.semantic-cache/`, like page vectors.

The cache is a pure derived artifact: safe to delete (it rebuilds), never
committed. Deletions in the corpus are handled implicitly — each call rewrites
the cache to exactly the rows it was asked about, so stale claims fall out.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

from ..paths import semantic_cache_dir
from ..fsatomic import exclusive_lock
from ..log import log
from .bundle import load_bundle, save_bundle
from . import embeddings


def _paths() -> tuple[Path, Path]:
    d = semantic_cache_dir()
    return d / "claims.npy", d / "claims_meta.json"


def _identity(row: dict) -> str:
    return f"{row['paper_stem']}·{row['section']}·{row['position']}"


def _text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


def _bundle_path() -> Path:
    return _paths()[0].with_suffix(".npz")


def _cached():
    return load_bundle(_bundle_path(), embeddings.DEFAULT_MODEL, row_fields=("id", "hash"))


def load_cached_claim_embeddings(rows: list[dict]):
    """Cache-only lookup; an unknown legacy pair is deliberately a miss."""
    if not _NUMPY or not rows:
        return None
    cached = _cached()
    if cached is None:
        return None
    vecs, meta = cached
    index = {m["id"]: (i, m["hash"]) for i, m in enumerate(meta["rows"])}
    covered, picked = [], []
    for i, row in enumerate(rows):
        hit = index.get(_identity(row))
        if hit is not None and hit[1] == _text_hash(row["text"]):
            covered.append(i)
            picked.append(hit[0])
    if not picked:
        return None
    out = vecs[picked].astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms, covered


def get_claim_embeddings(rows: list[dict]):
    """Embed misses and replace the cache with the requested claim set."""
    return _get(rows, union=False)


def warm_claim_embeddings(rows: list[dict]):
    """Embed only misses, merging with the latest cache at publication."""
    return _get(rows, union=True)


def _get(rows: list[dict], *, union: bool):
    if not _NUMPY or not rows:
        return None
    cached = _cached()
    vectors, meta = cached if cached else (None, {"rows": []})
    index = {m["id"]: (i, m["hash"]) for i, m in enumerate(meta["rows"])}
    reuse, missing = {}, []
    for i, row in enumerate(rows):
        hit = index.get(_identity(row))
        if hit is not None and hit[1] == _text_hash(row["text"]):
            reuse[i] = hit[0]
        else:
            missing.append(i)
    fresh = None
    if missing:
        if not embeddings.is_available():
            return None
        fresh = embeddings.embed_texts([rows[i]["text"] for i in missing])
        if fresh is None or fresh.ndim != 2 or len(fresh) != len(missing):
            return None
    if fresh is not None and vectors is not None and fresh.shape[1] != vectors.shape[1]:
        # The backend changed dimension without changing its configured name.
        fresh = embeddings.embed_texts([row["text"] for row in rows])
        if fresh is None:
            return None
        missing, reuse = list(range(len(rows))), {}
    dim = fresh.shape[1] if fresh is not None else vectors.shape[1]
    out = np.zeros((len(rows), dim), dtype=np.float32)
    for i, cached_i in reuse.items():
        out[i] = vectors[cached_i]
    for i, row_i in enumerate(missing):
        out[row_i] = fresh[i]
    try:
        with exclusive_lock(_bundle_path()):
            entries = {}
            if union:
                latest = _cached()
                if latest is not None and latest[0].shape[1] == dim:
                    entries = {m["id"]: (m, latest[0][i])
                               for i, m in enumerate(latest[1]["rows"])}
            for i, row in enumerate(rows):
                ident = _identity(row)
                entries[ident] = ({"id": ident, "hash": _text_hash(row["text"])}, out[i])
            save_bundle(_bundle_path(), np.stack([v for m, v in entries.values()]), {
                "model": embeddings.DEFAULT_MODEL, "dim": dim,
                "rows": [m for m, v in entries.values()],
            })
    except OSError as exc:
        log(f"claim embedding cache write failed: {exc}", tag="cache")
    return out
