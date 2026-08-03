"""Page-level semantic embedding store (A-Mem-inspired).

Complements the Tantivy BM25 index in `tantivy_.py`: a dense per-page vector
keyed by `category/stem`, used for See-Also, ingest-time link/evolution
candidate selection, and hybrid retrieval.

Indexed text per page mirrors A-Mem's `concat(c, K, G, X)` — title, summary,
key contributions, tags, keywords. The model is shared with the claim-grading
bi-encoder (`index.embeddings`) so we don't pay a second 133 MB load.

Storage layout under `.semantic-cache/`:
  pages.npy         float32, shape (N, dim), L2-normalized rows
  pages_meta.json   {model, dim, built_at, rows: [{key, stem, category,
                    page_type, title, content_hash}, ...]}

Row order in `pages.npy` matches `rows` in `pages_meta.json`. `content_hash`
lets a future incremental builder skip re-embedding unchanged pages — for now
we always rebuild from scratch (matches the Tantivy contract).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import embeddings as semantic
from ..paths import semantic_cache_dir
from ..wiki import Page, extract_section, read_pages
from ..log import log

PAGES_NPY = "pages.npy"
PAGES_META = "pages_meta.json"


@dataclass
class PageHit:
    key: str                 # "category/stem"
    stem: str
    category: str
    page_type: str
    title: str
    score: float             # cosine similarity in [0, 1]


def page_index_text(p: Page) -> str:
    """The text we embed for a page.

    A-Mem's note encoding concatenates content + keywords + tags + context
    before embedding. We do the analog over the wiki page's high-signal
    fields — the full body is too noisy at the page level (template-driven
    sections, wikilink syntax, etc.) and dilutes the cosine signal.
    """
    parts: list[str] = []
    title = p.fm.get("title", "").strip()
    if title:
        parts.append(title)

    summary = extract_section(p.body, "Summary").strip()
    if summary:
        parts.append(summary)

    # Key Contributions only exists on paper pages — synthesis pages just
    # contribute their summary instead.
    contribs = extract_section(p.body, "Key Contributions").strip()
    if contribs:
        parts.append(contribs)

    keywords = p.str_field("keywords").strip()
    if keywords:
        parts.append(keywords)

    tags = p.str_field("tags").strip()
    if tags:
        parts.append(tags)

    return "\n\n".join(parts)


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_index(pages: list[Page] | None = None) -> dict | None:
    """Rebuild the page-level semantic index from `wiki/`.

    Returns a small summary dict on success, or None if the embedding model
    is unavailable. Empty wiki returns a summary with `n_pages: 0`.
    """
    if not semantic.is_available():
        log("embedding model unavailable; skipping", tag="semantic-pages")
        return None

    if pages is None:
        pages = read_pages()

    rows: list[dict] = []
    texts: list[str] = []
    for p in pages:
        text = page_index_text(p)
        if not text.strip():
            # Skip pages with nothing indexable. Sparse stubs would otherwise
            # cluster around (0, ...) and pollute KNN.
            continue
        rows.append({
            "key": p.key,
            "stem": p.stem,
            "category": p.category,
            "page_type": p.page_type,
            "title": p.fm.get("title", ""),
            "content_hash": _content_hash(text),
        })
        texts.append(text)

    out_dir = semantic_cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not texts:
        # Wipe any stale state so we don't leave a half-built index that
        # mismatches the wiki.
        _write_empty(out_dir)
        return {"n_pages": 0, "model": semantic.DEFAULT_MODEL, "dim": 0}

    embeddings = semantic.embed_texts(texts)
    if embeddings is None or embeddings.shape[0] != len(texts):
        log("embedding pass failed", tag="semantic-pages")
        return None

    np.save(out_dir / PAGES_NPY, embeddings)
    meta = {
        "model": semantic.DEFAULT_MODEL,
        "dim": int(embeddings.shape[1]),
        "built_at": int(time.time()),
        "rows": rows,
    }
    (out_dir / PAGES_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"n_pages": len(rows), "model": meta["model"], "dim": meta["dim"]}


def _write_empty(out_dir: Path) -> None:
    np.save(out_dir / PAGES_NPY, np.zeros((0, 0), dtype=np.float32))
    (out_dir / PAGES_META).write_text(json.dumps({
        "model": semantic.DEFAULT_MODEL,
        "dim": 0,
        "built_at": int(time.time()),
        "rows": [],
    }, indent=2), encoding="utf-8")


def _load() -> tuple[np.ndarray, list[dict]] | None:
    out_dir = semantic_cache_dir()
    npy = out_dir / PAGES_NPY
    meta = out_dir / PAGES_META
    if not npy.exists() or not meta.exists():
        return None
    rows = json.loads(meta.read_text(encoding="utf-8")).get("rows", [])
    arr = np.load(npy)
    if arr.shape[0] != len(rows):
        # Index is corrupt or partially written — caller should rebuild.
        return None
    return arr, rows


def load_index() -> tuple[np.ndarray, list[dict]] | None:
    """Public loader for the page embedding store.

    Returns (embeddings_matrix, row_meta_list) where:
      - embeddings_matrix is shape (N, dim), L2-normalized
      - row_meta_list[i] = {key, stem, category, page_type, title, content_hash}
        for the i-th row of the matrix

    Returns None when the index isn't built or is corrupt. Callers that need
    pairwise similarity (e.g. cluster detection in `candidates synthesis`) read
    the raw matrix directly instead of going through `query_text` per page —
    `arr @ arr.T` computes all pairwise cosines in one matrix multiply.
    """
    return _load()


def index_exists() -> bool:
    """Return True iff a built page-level index can be loaded."""
    return _load() is not None


def query_text(text: str, k: int = 8,
               page_types: tuple[str, ...] | None = None,
               exclude_keys: frozenset[str] = frozenset()) -> list[PageHit]:
    """Top-k pages most similar to free-form `text`.

    `page_types` filters by `type:` frontmatter (e.g. `("synthesis",)` for
    the evolution-target shortlist).
    `exclude_keys` drops specific `category/stem` keys (e.g. the seed page
    itself when called via `query_stem`).
    """
    loaded = _load()
    if loaded is None:
        return []
    arr, rows = loaded
    if arr.shape[0] == 0:
        return []

    q = semantic.embed_texts([text])
    if q is None or q.shape[0] == 0:
        return []

    sims = arr @ q[0]                         # (N,)
    order = np.argsort(-sims)                 # high to low

    out: list[PageHit] = []
    for idx in order:
        row = rows[int(idx)]
        if row["key"] in exclude_keys:
            continue
        if page_types is not None and row["page_type"] not in page_types:
            continue
        out.append(PageHit(
            key=row["key"],
            stem=row["stem"],
            category=row["category"],
            page_type=row["page_type"],
            title=row["title"],
            score=float(max(0.0, min(1.0, sims[int(idx)]))),
        ))
        if len(out) >= k:
            break
    return out


def query_stem(key: str, k: int = 8,
               page_types: tuple[str, ...] | None = None) -> list[PageHit]:
    """Top-k neighbors of an existing page identified by `category/stem`.

    The seed page is excluded from the result. Returns `[]` if the seed isn't
    indexed (e.g., empty page or built before the page was added).
    """
    loaded = _load()
    if loaded is None:
        return []
    arr, rows = loaded
    if arr.shape[0] == 0:
        return []

    seed_idx: int | None = None
    for i, row in enumerate(rows):
        if row["key"] == key:
            seed_idx = i
            break
    if seed_idx is None:
        return []

    sims = arr @ arr[seed_idx]                # cosine since rows are normalized
    sims[seed_idx] = -1.0                     # mask out self

    order = np.argsort(-sims)
    out: list[PageHit] = []
    for idx in order:
        i = int(idx)
        if sims[i] <= 0:
            break
        row = rows[i]
        if page_types is not None and row["page_type"] not in page_types:
            continue
        out.append(PageHit(
            key=row["key"],
            stem=row["stem"],
            category=row["category"],
            page_type=row["page_type"],
            title=row["title"],
            score=float(max(0.0, min(1.0, sims[i]))),
        ))
        if len(out) >= k:
            break
    return out
