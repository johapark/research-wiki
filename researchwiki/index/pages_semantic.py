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
import re
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import embeddings as semantic
from ..paths import semantic_cache_dir
from ..wiki import Page, extract_section, read_pages, strip_non_prose
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


#: Sections that carry a page's substance, per page type. `page_index_text`
#: extracted only `Summary` + `Key Contributions` — names that exist on paper
#: pages and on **no other type** — so every synthesis, idea and concept page was
#: embedded on its title alone. Measured 2026-08-06, median embedded length:
#: paper 2539 chars, synthesis 51, idea 64, concept 23 (minimum 15). Their real
#: H2s were never read.
_INDEX_SECTIONS: dict[str, tuple[str, ...]] = {
    "paper":      ("Summary", "Key Contributions"),
    "commentary": ("Summary", "Key Contributions"),
    "concept":    ("Definition", "How it appears across the corpus",
                   "Cross-domain connections"),
    "idea":       ("Verdict", "Background", "Opportunities"),
    "synthesis":  ("Short answer", "Question", "Evidence from the wiki", "Summary"),
}
_INDEX_SECTIONS_DEFAULT = ("Summary", "Key Contributions", "Definition")

#: Page types whose `tags:` carry a topical vocabulary worth embedding.
#:
#: The split is not cosmetic — the field is used for two different jobs. On paper
#: pages it is provenance: `ingested-via-agent` accounts for 334 of 391 pages'
#: tags, a token identical across three quarters of the corpus that adds no
#: discrimination while occupying room in every vector. On concept / idea /
#: synthesis pages it is the only keyword-like field they have, because
#: `find_missing_keywords` exempts exactly those three directories — so that is
#: where a real vocabulary accumulated (`dna-foundation-model`, `pangenome`,
#: `deep-mutational-scanning`, `ldl-c`), at 2.0 / 7.9 / 3.5 tags per page against
#: the paper pages' 1.9 that are mostly one provenance marker.
_TAGS_CARRY_SIGNAL = frozenset({"concept", "idea", "synthesis"})

#: Tags that describe the pipeline rather than the paper. Stripped even on the
#: types above, so a stray provenance marker can't dilute a small vector.
_FRAMEWORK_TAGS = frozenset({
    "ingested-via-agent", "migrated", "synthesis", "concept", "idea",
    "whitepaper", "guidance", "protocol", "book", "paper", "commentary",
})


def page_index_text(p: Page) -> str:
    """The text we embed for a page.

    A-Mem's note encoding concatenates content + keywords + tags + context before
    embedding. We do the analog over the wiki page's high-signal fields — the full
    body is too noisy at the page level (template-driven sections, wikilink
    syntax) and dilutes the cosine signal.

    Which fields count is **type-dependent**, and getting that wrong is how
    synthesis / idea / concept pages came to be embedded on nothing but their
    titles: the section names were paper-page names, and the one field that did
    reach them, `tags:`, was removed earlier the same day on a corpus-wide
    argument that only held for papers. See `_INDEX_SECTIONS` and
    `_TAGS_CARRY_SIGNAL` for the measurements.

    Section names are matched by type with a permissive fallback, so a page whose
    H2s don't match its declared type still contributes whatever it does have
    rather than silently reducing to a title.

    **Order is load-bearing.** The bi-encoder truncates at `max_seq_length` 512
    tokens, roughly 2000 characters, so anything past that is not embedded at all.
    Each type's list therefore leads with its own tl;dr — `Verdict` for an idea,
    `Short answer` for a synthesis, `Summary` for a paper — and the longer
    supporting sections trail behind where truncation costs least. An idea page
    assembles ~11.8k characters and only its first fifth is read; that is fine,
    and it is fine *because* of the ordering.
    """
    parts: list[str] = []
    page_type = str(p.fm.get("type") or "paper").strip().strip("\"'")

    title = p.fm.get("title", "").strip()
    if title:
        parts.append(title)

    wanted = _INDEX_SECTIONS.get(page_type, _INDEX_SECTIONS_DEFAULT)
    seen: set[str] = set()
    for name in (*wanted, *_INDEX_SECTIONS_DEFAULT):
        if name in seen:
            continue
        seen.add(name)
        section = extract_section(p.body, name).strip()
        if section:
            parts.append(section)

    # Synthesis pages have no mandated H2 contract (unlike idea and concept, whose
    # section order CLAUDE.md fixes), so name matching cannot be exhaustive there:
    # `synthesis/dsb-free-editing-axis` uses "The axis" / "Positions on the axis",
    # and `synthesis/suggested-additions` uses "Priority 1 — …". Both reduced to a
    # bare title under name matching alone. Falling back to the body is safe
    # because the embedder truncates anyway.
    if len(parts) <= 1:
        body = _drop_section(p.body, "References")
        prose = strip_non_prose(body).strip()
        if prose:
            parts.append(prose)

    keywords = p.str_field("keywords").strip()
    if keywords:
        parts.append(keywords)

    if page_type in _TAGS_CARRY_SIGNAL:
        tags = [
            tg for tg in _tag_list(p.fm.get("tags"))
            if tg.lower() not in _FRAMEWORK_TAGS
        ]
        if tags:
            parts.append(", ".join(tags))

    return "\n\n".join(parts)


def _drop_section(body: str, name: str) -> str:
    """`body` with one `## name` section removed, heading included.

    Used to keep a References block of bare `[^id]: [[stem]]` footnotes out of the
    whole-body fallback, where it would contribute a wall of stems rather than
    prose.
    """
    m = re.search(rf"^## {re.escape(name)}\s*$", body, re.MULTILINE)
    if not m:
        return body
    nxt = re.search(r"^## ", body[m.end():], re.MULTILINE)
    end = m.end() + nxt.start() if nxt else len(body)
    return body[:m.start()] + body[end:]


def _tag_list(raw) -> list[str]:
    """`tags:` as a list of strings, tolerating the inline-YAML string form."""
    if isinstance(raw, str):
        return [x.strip() for x in raw.strip().strip("[]").split(",") if x.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


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
