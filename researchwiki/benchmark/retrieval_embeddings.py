"""Model-aware encoders and fresh, isolated retrieval benchmark caches.

Harrier's packaged sentence-transformers configuration supplies its query
instruction and last-token pooling. Documents receive no instruction. This
bridge never changes the production encoder used for claim grading.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..db import get_connection
from ..errors import EnvironmentFailure
from ..index.bundle import load_bundle, save_bundle
from ..log import log
from ..paths import wiki_root


BGE_MODEL = "BAAI/bge-small-en-v1.5"
HARRIER_MODELS = {
    "microsoft/harrier-oss-v1-270m",
    "microsoft/harrier-oss-v1-0.6b",
}
MODEL_ALIASES = {
    "bge-small": BGE_MODEL,
    "harrier-270m": "microsoft/harrier-oss-v1-270m",
    "harrier-0.6b": "microsoft/harrier-oss-v1-0.6b",
}


class EmbeddingUnavailable(EnvironmentFailure):
    """A failed encoder must not be reported as zero retrieval quality."""


@dataclass(frozen=True)
class ModelOptions:
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int | None = None
    max_seq_length: int | None = None
    cache_folder: str | None = None


_OPTIONS: dict[str, ModelOptions] = {}
_MODEL_CACHE: dict[tuple, object] = {}
_LOAD_MS: dict[tuple, float] = {}
_CACHE_EVENTS: list[dict] = []


def resolve_model(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def configure_model(model_name: str, **kwargs) -> None:
    options = ModelOptions(**kwargs)
    if options.batch_size is not None and options.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if options.max_seq_length is not None and options.max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")
    if options.dtype not in ("auto", "float32", "bfloat16"):
        raise ValueError("dtype must be auto, float32, or bfloat16")
    _OPTIONS[resolve_model(model_name)] = options


def _model_key(model_name: str, trust_remote_code: bool) -> tuple:
    model_id = resolve_model(model_name)
    return model_id, _OPTIONS.get(model_id, ModelOptions()), trust_remote_code


@contextmanager
def _model_download_mode(model_id: str):
    """BGE's automatic offline optimization must not block a new candidate.

    An explicit user HF_HUB_OFFLINE value remains authoritative. Only the
    framework's own BGE-cache optimization is temporarily lifted, including
    the Hub constant that was captured when that library was imported.
    """
    from ..index import embeddings
    automatic = (embeddings._AUTOMATIC_OFFLINE_MODE and model_id != BGE_MODEL
                 and os.environ.get("HF_HUB_OFFLINE") == "1")
    if not automatic:
        yield
        return
    from huggingface_hub import constants
    previous = constants.HF_HUB_OFFLINE
    os.environ["HF_HUB_OFFLINE"] = "0"
    constants.HF_HUB_OFFLINE = False
    try:
        yield
    finally:
        os.environ["HF_HUB_OFFLINE"] = "1"
        constants.HF_HUB_OFFLINE = previous


def _load_model(model_name: str, *, trust_remote_code: bool = False):
    key = _model_key(model_name, trust_remote_code)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model_id, options, _ = key
    start = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
        kwargs = {"device": options.device, "trust_remote_code": trust_remote_code}
        if options.cache_folder is not None:
            kwargs["cache_folder"] = options.cache_folder
        with _model_download_mode(model_id):
            model = SentenceTransformer(model_id, **kwargs)
        if options.dtype == "float32":
            model.float()
        elif options.dtype == "bfloat16":
            model.bfloat16()
        if options.max_seq_length is not None:
            model.max_seq_length = min(model.max_seq_length, options.max_seq_length)
        if model_id in HARRIER_MODELS and "web_search_query" not in model.prompts:
            raise ValueError("Harrier checkpoint is missing its web_search_query prompt")
    except Exception as exc:
        hint = " Install current dependencies with pip install -e '.[harrier]'." if model_id in HARRIER_MODELS else ""
        log(f"failed to load model {model_id!r}: {exc}{hint}", tag="retrieval")
        return None
    _MODEL_CACHE[key] = model
    _LOAD_MS[key] = (time.perf_counter() - start) * 1000
    return model


def _encode_kwargs(model_name: str, *, role: str, prefix: str) -> dict:
    model_id = resolve_model(model_name)
    options = _OPTIONS.get(model_id, ModelOptions())
    kwargs = {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
        "batch_size": options.batch_size or (8 if model_id in HARRIER_MODELS else 32),
    }
    if prefix:
        # Explicit prefixes override the automatic instruction, never stack it.
        kwargs["prompt"] = ""
    elif model_id in HARRIER_MODELS:
        if role == "query":
            kwargs["prompt_name"] = "web_search_query"
        else:
            kwargs["prompt"] = ""
    return kwargs


def _embed_with(texts: list[str], model_name: str, *, trust_remote_code: bool = False,
                prefix: str = "", role: str = "document") -> np.ndarray | None:
    model = _load_model(model_name, trust_remote_code=trust_remote_code)
    if model is None:
        return None
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    inputs = [prefix + text for text in texts] if prefix else texts
    try:
        vectors = model.encode(inputs, **_encode_kwargs(model_name, role=role, prefix=prefix))
    except Exception as exc:
        raise EmbeddingUnavailable(f"encoding failed for {resolve_model(model_name)}: {exc}") from exc
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise EmbeddingUnavailable(f"{resolve_model(model_name)} returned invalid embeddings")
    return vectors


def model_details(model_name: str, *, trust_remote_code: bool = False) -> dict:
    model = _load_model(model_name, trust_remote_code=trust_remote_code)
    if model is None:
        raise EmbeddingUnavailable(f"cannot load {resolve_model(model_name)}; check the model cache or network")
    model_id = resolve_model(model_name)
    options = _OPTIONS.get(model_id, ModelOptions())
    config = model[0].auto_model.config
    parameters = list(model.parameters())
    return {
        "model": model_id,
        "revision": getattr(config, "_commit_hash", None),
        "dimension": (model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension")
                      else model.get_sentence_embedding_dimension()),
        "max_seq_length": model.max_seq_length,
        "device": options.device,
        "dtype": str(parameters[0].dtype) if parameters else None,
        "parameter_count": sum(p.numel() for p in parameters),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "query_prompt_name": "web_search_query" if model_id in HARRIER_MODELS else None,
        "batch_size": _encode_kwargs(model_name, role="document", prefix="")["batch_size"],
        "load_ms": _LOAD_MS[_model_key(model_name, trust_remote_code)],
    }


def cache_dir_for(model_name: str) -> Path:
    model_id = resolve_model(model_name)
    slug = model_id.split("/")[-1]
    digest = hashlib.sha256(model_id.encode()).hexdigest()[:12]
    return wiki_root() / f".retrieval-cache-{slug}-{digest}"


def _all_db_claims() -> list[dict]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT paper_stem, section, position, text FROM claims "
            "WHERE is_cross_ref = 0 ORDER BY paper_stem, section, position"
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _page_inputs() -> tuple[list[dict], list[str]]:
    from ..index.pages_semantic import page_index_text
    from ..wiki import read_pages
    rows, texts = [], []
    for page in sorted(read_pages(), key=lambda p: p.key):
        if page.page_type in ("meta", "dashboard"):
            continue
        text = page_index_text(page)
        if not text.strip():
            continue
        rows.append({"key": page.key, "stem": page.stem, "category": page.category,
                     "page_type": page.page_type, "title": str(page.fm.get("title") or ""),
                     "content_hash": hashlib.sha256(text.encode()).hexdigest()})
        texts.append(text)
    return rows, texts


def corpus_fingerprint() -> dict:
    claims = _all_db_claims()
    pages, _ = _page_inputs()
    payload = json.dumps({"claims": claims, "pages": pages}, sort_keys=True, ensure_ascii=False)
    return {"sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "claims": len(claims), "pages": len(pages)}


def _cache_signature(model_name: str, doc_prefix: str, trust_remote_code: bool) -> dict:
    details = model_details(model_name, trust_remote_code=trust_remote_code)
    return {**{k: details[k] for k in ("model", "revision", "dimension", "max_seq_length", "dtype")},
            "doc_prefix": doc_prefix, "encoder_contract": 1}


def _production_vectors(kind: str, rows: list[dict], texts: list[str], signature: dict):
    # Only the exact default BGE representation can seed a benchmark. Read-only:
    # Harrier and all instruction/context overrides always build their own cache.
    if (signature["model"] != BGE_MODEL or signature["doc_prefix"]
            or signature["max_seq_length"] != 512 or signature["dtype"] != "torch.float32"):
        return None
    if kind == "claims":
        from ..index.claim_embeddings import load_cached_claim_embeddings
        cached = load_cached_claim_embeddings(rows)
        if cached is not None and cached[1] == list(range(len(rows))):
            return cached[0]
    else:
        from ..index.pages_semantic import _load, _content_hash
        cached = _load()
        if cached is not None:
            vectors, metadata = cached
            by_key = {row["key"]: i for i, row in enumerate(metadata)}
            picked = []
            for row, text in zip(rows, texts):
                idx = by_key.get(row["key"])
                if idx is None or metadata[idx].get("content_hash") != _content_hash(text):
                    return None
                picked.append(idx)
            return vectors[picked]
    return None


def _compatible_context(cached, rows: list[dict], texts: list[str], signature: dict,
                        model_name: str, trust_remote_code: bool) -> bool:
    """Reuse an untruncated known-model cache after changing only the token cap.

    Claims are short. A context-control experiment should not re-encode all of
    them when neither cap truncates any input; verify that with the tokenizer.
    """
    if cached is None or cached[1]["rows"] != rows:
        return False
    old = dict(cached[1].get("signature", {}))
    new = dict(signature)
    old_limit = old.pop("max_seq_length", None)
    new_limit = new.pop("max_seq_length")
    if (old != new or not isinstance(old_limit, int)
            or signature["model"] not in HARRIER_MODELS | {BGE_MODEL}):
        return False
    model = _load_model(model_name, trust_remote_code=trust_remote_code)
    if model is None:
        return False
    prefix = signature["doc_prefix"]
    inputs = [prefix + text for text in texts] if prefix else texts
    tokenized = model.tokenizer(inputs, truncation=False, verbose=False)
    return all(len(tokens) <= min(old_limit, new_limit) for tokens in tokenized["input_ids"])


def _load_or_build(kind: str, model_name: str, rows: list[dict], texts: list[str], *,
                   trust_remote_code: bool, doc_prefix: str):
    if not rows:
        raise EmbeddingUnavailable(f"no {kind} to benchmark; build the wiki and claims DB first")
    signature = _cache_signature(model_name, doc_prefix, trust_remote_code)
    path = cache_dir_for(model_name) / f"{kind}.npz"
    cached = load_bundle(path, signature["model"])
    source = "benchmark_cache"
    elapsed_ms = 0.0
    if cached is not None and cached[1].get("signature") == signature and cached[1]["rows"] == rows:
        vectors = cached[0]
    else:
        if _compatible_context(cached, rows, texts, signature, model_name, trust_remote_code):
            vectors = cached[0]
            source = "context_compatible_cache"
        else:
            vectors = _production_vectors(kind, rows, texts, signature)
            source = "production_cache" if vectors is not None else "encoded"
        if vectors is None:
            log(f"encoding {len(texts)} {kind} with {signature['model']}", tag="retrieval")
            start = time.perf_counter()
            vectors = _embed_with(texts, model_name, trust_remote_code=trust_remote_code,
                                  prefix=doc_prefix, role="document")
            elapsed_ms = (time.perf_counter() - start) * 1000
        if vectors is None:
            raise EmbeddingUnavailable(f"cannot embed {kind} with {signature['model']}")
        save_bundle(path, vectors, {"model": signature["model"], "dim": int(vectors.shape[1]),
                                   "signature": signature, "rows": rows, "built_at": int(time.time()),
                                   "build_source": source, "encode_ms": elapsed_ms})
    _CACHE_EVENTS.append({"model": signature["model"], "kind": kind, "rows": len(rows),
                          "source": source, "encode_ms": elapsed_ms})
    return vectors, rows


def _load_or_build_claim_cache(model_name: str, *, trust_remote_code: bool = False,
                               doc_prefix: str = ""):
    rows = _all_db_claims()
    return _load_or_build("claims", model_name, rows, [r["text"] for r in rows],
                          trust_remote_code=trust_remote_code, doc_prefix=doc_prefix)


def _load_or_build_page_cache(model_name: str, *, trust_remote_code: bool = False,
                              doc_prefix: str = ""):
    rows, texts = _page_inputs()
    return _load_or_build("pages", model_name, rows, texts,
                          trust_remote_code=trust_remote_code, doc_prefix=doc_prefix)
