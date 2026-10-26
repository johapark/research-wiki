"""Page-level semantic embedding store (A-Mem-inspired).

Complements the Tantivy BM25 index in `tantivy_.py`: a dense per-page vector
keyed by `category/stem`, used for See-Also, ingest-time link/evolution
candidate selection, and hybrid retrieval.

Indexed text per page mirrors A-Mem's `concat(c, K, G, X)` — title, summary,
key contributions, tags, keywords. The model is shared with the claim-grading
bi-encoder (`index.embeddings`) so we don't pay a second 133 MB load.

Storage under `.semantic-cache/`: `pages.npz` contains the vector matrix and
row-aligned JSON metadata (model, dimension, content hashes). Publication is one
atomic replacement. Legacy matrix/JSON pairs are cache misses. `pages.stale`
makes skipped embeddings unavailable until a successful refresh.

"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import embeddings as semantic
from ..evidence import EvidenceScope
from ..paths import semantic_cache_dir
from ..wiki import Page, extract_section, read_pages, strip_non_prose
from ..log import log
from ..fsatomic import exclusive_lock, write_text_atomic
from .bundle import load_bundle, save_bundle

PAGES_BUNDLE = "pages.npz"
PAGES_STALE = "pages.stale"
PAGES_NPY = "pages.npy"  # Legacy files are not trusted.
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
    "proposal":   ("Question", "Provisional thesis or hypothesis",
                   "Why this connection matters", "Evidence and reasoning",
                   "Transfer mapping", "Decisive uncertainty"),
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
_TAGS_CARRY_SIGNAL = frozenset({"concept", "idea", "synthesis", "proposal"})

#: Tags that describe the pipeline rather than the paper. Stripped even on the
#: types above, so a stray provenance marker can't dilute a small vector.
_FRAMEWORK_TAGS = frozenset({
    "ingested-via-agent", "migrated", "synthesis", "concept", "idea", "proposal",
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

    title = _frontmatter_text(p.fm.get("title"))
    if title:
        parts.append(title)

    wanted = _INDEX_SECTIONS.get(page_type, _INDEX_SECTIONS_DEFAULT)
    seen: set[str] = set()
    matched_a_section = False
    for name in (*wanted, *_INDEX_SECTIONS_DEFAULT):
        if name in seen:
            continue
        seen.add(name)
        section = extract_section(p.body, name).strip()
        if section:
            parts.append(section)
            matched_a_section = True

    # Synthesis pages have no mandated H2 contract (unlike idea and concept, whose
    # section order CLAUDE.md fixes), so name matching cannot be exhaustive there:
    # `synthesis/dsb-free-editing-axis` uses "The axis" / "Positions on the axis",
    # and `synthesis/suggested-additions` uses "Priority 1 — …". Both reduced to a
    # bare title under name matching alone. Falling back to the body is safe
    # because the embedder truncates anyway.
    # Gate on "did any section match", NOT on `len(parts)`. Counting parts made
    # the title load-bearing: a page with no `title:` and one matched section has
    # len(parts) == 1, so the fallback also fired and the section's text was
    # embedded twice.
    if not matched_a_section:
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


#: Below this many characters, a page's embedded text is too thin to retrieve on.
#:
#: Chosen from the measured gap rather than by feel. After sections were made
#: type-aware (2026-08-06) the shortest legitimate page embeds 726 chars
#: (whitepaper; paper 1399, commentary 827, synthesis 772, concept 3832, idea
#: 5500). While the type split was broken, the affected pages embedded 15-64
#: chars. So anything between ~70 and ~700 separates the two states, and 200
#: sits there with ~3.6x headroom under the real minimum — loose enough not to
#: nag about a genuinely terse page, tight enough that a collapse cannot hide.
INDEX_TEXT_FLOOR = 200


def thin_index_reason(p: Page, text: str | None = None) -> str | None:
    """Why this page's embedded text is too thin to retrieve on, else None.

    The guard for a failure that is otherwise **silent**: on 2026-08-06 a
    section-name mismatch reduced every synthesis, idea and concept page to its
    title, and nothing noticed. `lint` had no view of the embedding, the unit
    tests used synthetic pages, the check reached for (`stale_by_content`) runs on
    BM25 and so could not reflect it, and the retrieval benchmark's fixtures are
    paper-anchored by schema with no runner. A page that embeds only its title is
    wrong on its face, needs no ground truth to detect, and is exactly what was
    missed.

    Deliberately calls `page_index_text` rather than re-deriving the parts, so it
    can never disagree with what is actually embedded. Pass `text` when the caller
    already has it (`build_index` does) to skip the second assembly.
    """
    text = page_index_text(p) if text is None else text
    body = text.strip()
    if not body:
        return "empty — nothing indexable"
    title = str(p.fm.get("title") or "").strip()
    if title and body == title:
        page_type = str(p.fm.get("type") or "paper").strip().strip("\"'")
        wanted = _INDEX_SECTIONS.get(page_type, _INDEX_SECTIONS_DEFAULT)
        return (f"title-only — no section matched for type={page_type!r} "
                f"(looked for {', '.join(wanted)})")
    if len(body) < INDEX_TEXT_FLOOR:
        return f"only {len(body)} chars (floor {INDEX_TEXT_FLOOR})"
    return None


def _tag_list(raw) -> list[str]:
    """`tags:` as a list of strings, tolerating the inline-YAML string form."""
    if isinstance(raw, str):
        return [x.strip() for x in raw.strip().strip("[]").split(",") if x.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _frontmatter_text(value) -> str:
    """A frontmatter scalar as display text: `None` → "", `2021` → "2021"."""
    return "" if value is None else str(value).strip()


def _index_row(p: Page, text: str) -> dict:
    """Row-aligned metadata for one embedded page; equality means "vector current".

    Every field is written as a string because `load_bundle` rejects the whole
    bundle when any row field is not one. Frontmatter is hand-edited: a blank
    `type:` loads as `None` and an unquoted `title: 2021` as an int, and either
    used to make the index unreadable — `index_exists()` false for every
    caller, and each later upsert re-embedding the entire corpus. A blank type
    is recorded as `paper`, the same default `page_index_text` embeds it under;
    `lint` reports it as `missing_type`.
    """
    return {
        "key": p.key,
        "stem": p.stem,
        "category": p.category,
        "page_type": _frontmatter_text(p.fm.get("type")) or "paper",
        "title": _frontmatter_text(p.fm.get("title")),
        "content_hash": _content_hash(text),
    }


def prepare_vectors(pages: list[Page], *, full: bool) -> dict[str, np.ndarray]:
    """Embed, holding no lock, the page texts the next build or upsert needs.

    Embedding is the slow part of page maintenance — a model load, then seconds
    to minutes of encoding on a full rebuild — so callers run this before taking
    the corpus guard and pass the result to `build_index`/`upsert_pages` as
    `prepared`. Vectors are keyed by the content hash of the exact text they
    encode, so one can only be used for that text: a page edited in between
    misses and is embedded inside the lock instead, and no vector is published
    for text that is no longer on disk.

    A full build needs every page. An upsert needs only rows the current
    bundle lacks — or every page when the bundle is unusable or stale, since
    that upsert then reconciles the whole wiki. Returns {} when the model is
    unavailable; the locked step reports that.
    """
    if not semantic.is_available():
        return {}
    have: dict[str, dict] = {}
    if not full:
        loaded = _load(allow_stale=True)
        if loaded is None or (semantic_cache_dir() / PAGES_STALE).exists():
            pages = read_pages()
        if loaded is not None:
            have = {row["stem"]: row for row in loaded[1]}
    need: dict[str, str] = {}
    for page in pages:
        text = page_index_text(page)
        if text.strip() and (full or have.get(page.stem) != _index_row(page, text)):
            need.setdefault(_content_hash(text), text)
    if not need:
        return {}
    vectors = semantic.embed_texts(list(need.values()))
    if vectors is None or vectors.ndim != 2 or vectors.shape[0] != len(need):
        return {}
    return dict(zip(need, vectors))


def _embed(texts: list[str], prepared: dict[str, np.ndarray] | None) -> np.ndarray | None:
    """Vectors for `texts`, taking any `prepare_vectors` already computed."""
    if not prepared:
        return semantic.embed_texts(texts)
    hashes = [_content_hash(text) for text in texts]
    missing = {h: text for h, text in zip(hashes, texts) if h not in prepared}
    fresh: dict[str, np.ndarray] = {}
    if missing:
        vectors = semantic.embed_texts(list(missing.values()))
        if vectors is None or vectors.ndim != 2 or vectors.shape[0] != len(missing):
            return None
        fresh = dict(zip(missing, vectors))
    return np.vstack([prepared.get(h, fresh.get(h)) for h in hashes]).astype(np.float32, copy=False)


def _locked(function):
    from functools import wraps
    @wraps(function)
    def call(*args, **kwargs):
        with exclusive_lock(semantic_cache_dir() / PAGES_BUNDLE):
            return function(*args, **kwargs)
    return call


@_locked
def build_index(pages: list[Page] | None = None, *,
                prepared: dict[str, np.ndarray] | None = None) -> dict | None:
    """Rebuild the page-level semantic index from `wiki/`.

    Returns a small summary dict on success, or None if the embedding model
    is unavailable. Empty wiki returns a summary with `n_pages: 0`.
    `prepared` holds vectors from `prepare_vectors`, used for matching text.
    """
    if not semantic.is_available():
        log("embedding model unavailable; skipping", tag="semantic-pages")
        return None

    if pages is None:
        pages = read_pages()

    rows: list[dict] = []
    texts: list[str] = []
    thin: list[dict] = []
    for p in pages:
        text = page_index_text(p)
        reason = thin_index_reason(p, text)
        if reason:
            thin.append({"key": p.key, "page_type": p.page_type, "reason": reason})
        if not text.strip():
            # Skip pages with nothing indexable. Sparse stubs would otherwise
            # cluster around (0, ...) and pollute KNN.
            continue
        rows.append(_index_row(p, text))
        texts.append(text)

    out_dir = semantic_cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not texts:
        # Wipe any stale state so we don't leave a half-built index that
        # mismatches the wiki.
        _write_empty(out_dir)
        return {"n_pages": 0, "model": semantic.DEFAULT_MODEL, "dim": 0, "thin": thin}

    embeddings = _embed(texts, prepared)
    if embeddings is None or embeddings.shape[0] != len(texts):
        log("embedding pass failed", tag="semantic-pages")
        return None

    meta = {
        "model": semantic.DEFAULT_MODEL,
        "dim": int(embeddings.shape[1]),
        "built_at": int(time.time()),
        "rows": rows,
    }
    save_bundle(out_dir / PAGES_BUNDLE, embeddings, meta)
    (out_dir / PAGES_STALE).unlink(missing_ok=True)
    return {"n_pages": len(rows), "model": meta["model"], "dim": meta["dim"],
            "thin": thin}


@_locked
def upsert_pages(pages: list[Page], *,
                 prepared: dict[str, np.ndarray] | None = None) -> dict | None:
    """Embed and upsert only changed pages in the existing semantic index.

    Rows are keyed by stem so a category move replaces the old key. Unchanged
    content hashes skip embedding. If the aligned cache is absent/corrupt, fall
    back to a full rebuild—the Markdown tree remains canonical.
    """
    if not semantic.is_available():
        log("embedding model unavailable; skipping", tag="semantic-pages")
        return None
    stale = (semantic_cache_dir() / PAGES_STALE).exists()
    loaded = _load(allow_stale=True)
    if loaded is None:
        result = build_index(prepared=prepared)
        if result is not None:
            result["mode"] = "rebuilt"
        return result

    arr, old_rows = loaded
    if stale:
        # A skipped full maintenance may have changed pages beyond this batch.
        # Reconcile the whole current row set, reusing unchanged vectors.
        pages = read_pages()
    by_stem = {p.stem: p for p in pages}
    if not by_stem:
        if stale:
            _write_empty(semantic_cache_dir())
            return {"n_pages": 0, "n_updated": len(old_rows), "n_embedded": 0,
                    "model": semantic.DEFAULT_MODEL, "dim": 0, "mode": "incremental", "thin": []}
        return {"n_pages": len(old_rows), "n_updated": 0, "n_embedded": 0,
                "model": semantic.DEFAULT_MODEL, "dim": int(arr.shape[1] if arr.ndim == 2 else 0),
                "mode": "incremental", "thin": []}

    old_by_stem = {row["stem"]: (i, row) for i, row in enumerate(old_rows)}
    pending: list[tuple[Page, str, dict]] = []
    thin: list[dict] = []
    removed_stems: set[str] = set(old_by_stem) - set(by_stem) if stale else set()
    for stem, page in by_stem.items():
        text = page_index_text(page)
        reason = thin_index_reason(page, text)
        if reason:
            thin.append({"key": page.key, "page_type": page.page_type, "reason": reason})
        if not text.strip():
            if stem in old_by_stem:
                removed_stems.add(stem)
            continue
        row = _index_row(page, text)
        prior = old_by_stem.get(stem)
        if prior is not None and prior[1] == row:
            continue
        pending.append((page, text, row))

    new_vectors = None
    if pending:
        new_vectors = _embed([text for _, text, _ in pending], prepared)
        if new_vectors is None or new_vectors.shape[0] != len(pending):
            log("incremental embedding pass failed", tag="semantic-pages")
            return None
        if arr.ndim != 2 or new_vectors.ndim != 2 or new_vectors.shape[1] != arr.shape[1]:
            # A cache produced by another embedding model/build cannot be
            # safely row-spliced. Rebuild every row under the active model.
            result = build_index(prepared=prepared)
            if result is not None:
                result["mode"] = "rebuilt"
            return result

    replaced = removed_stems | {page.stem for page, _, _ in pending}
    keep_indices = [i for i, row in enumerate(old_rows) if row["stem"] not in replaced]
    rows = [old_rows[i] for i in keep_indices]
    kept = arr[keep_indices] if keep_indices else np.zeros((0, arr.shape[1]), dtype=np.float32)
    if pending and new_vectors is not None:
        rows.extend(row for _, _, row in pending)
        embeddings = np.vstack([kept, new_vectors]) if kept.size else new_vectors
    else:
        embeddings = kept

    if not replaced:
        (semantic_cache_dir() / PAGES_STALE).unlink(missing_ok=True)
        return {"n_pages": len(rows), "n_updated": 0, "n_embedded": 0,
                "model": semantic.DEFAULT_MODEL, "dim": int(embeddings.shape[1] if embeddings.ndim == 2 else 0),
                "mode": "incremental", "thin": thin}

    out_dir = semantic_cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": semantic.DEFAULT_MODEL,
        "dim": int(embeddings.shape[1] if embeddings.ndim == 2 else 0),
        "built_at": int(time.time()),
        "rows": rows,
    }
    save_bundle(out_dir / PAGES_BUNDLE, embeddings, meta)
    (out_dir / PAGES_STALE).unlink(missing_ok=True)
    return {"n_pages": len(rows), "n_updated": len(replaced),
            "n_embedded": len(pending), "model": meta["model"],
            "dim": meta["dim"], "mode": "incremental", "thin": thin}


def _write_empty(out_dir: Path) -> None:
    save_bundle(out_dir / PAGES_BUNDLE, np.zeros((0, 0), dtype=np.float32), {
        "model": semantic.DEFAULT_MODEL, "dim": 0,
        "built_at": int(time.time()), "rows": [],
    })
    (out_dir / PAGES_STALE).unlink(missing_ok=True)


@_locked
def mark_stale() -> None:
    """Make skipped embeddings unavailable while retaining vectors for refresh."""
    write_text_atomic(semantic_cache_dir() / PAGES_STALE, "Run researchwiki reindex.\n")


@_locked
def drop_outdated(pages: list[Page], absent_stems: set[str]) -> None:
    """Keep the page index truthful after a page mutation, without a model.

    Rows for deleted pages and rows whose indexed text changed are dropped: an
    incomplete index is still correct, an outdated vector is not. Only a page
    that still has indexable text but lost its vector marks the index stale, so
    the next upsert reconciles it in full. A page that was never indexed — a
    fresh promote, which the ingest runner embeds next — leaves it available,
    and so does a back-link splice, which edits no indexed section.
    """
    out_dir = semantic_cache_dir()
    if (out_dir / PAGES_STALE).exists():
        return
    loaded = _load()
    if loaded is None:
        return
    arr, rows = loaded
    current = {}
    for page in pages:
        text = page_index_text(page)
        current[page.stem] = _index_row(page, text) if text.strip() else None
    drop = set(absent_stems)
    lost = False
    for row in rows:
        stem = row["stem"]
        if stem in current and current[stem] != row:
            drop.add(stem)
            lost = lost or current[stem] is not None
    keep = [i for i, row in enumerate(rows) if row["stem"] not in drop]
    if len(keep) != len(rows):
        kept = arr[keep] if keep else np.zeros((0, arr.shape[1] if arr.ndim == 2 else 0),
                                               dtype=np.float32)
        save_bundle(out_dir / PAGES_BUNDLE, kept, {
            "model": semantic.DEFAULT_MODEL,
            "dim": int(kept.shape[1]) if kept.ndim == 2 else 0,
            "built_at": int(time.time()),
            "rows": [rows[i] for i in keep],
        })
    if lost:
        # Written directly: this function already holds the bundle lock.
        write_text_atomic(out_dir / PAGES_STALE, "Run researchwiki reindex.\n")


def _load(*, allow_stale=False) -> tuple[np.ndarray, list[dict]] | None:
    if not allow_stale and (semantic_cache_dir() / PAGES_STALE).exists():
        return None
    loaded = load_bundle(semantic_cache_dir() / PAGES_BUNDLE, semantic.DEFAULT_MODEL,
                         row_fields=("key", "stem", "category", "page_type", "title", "content_hash"))
    return (loaded[0], loaded[1]["rows"]) if loaded is not None else None


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
               exclude_keys: frozenset[str] = frozenset(), *,
               scope: EvidenceScope | None = None) -> list[PageHit]:
    """Top-k pages most similar to free-form `text`.

    `page_types` filters by `type:` frontmatter (e.g. `("synthesis",)` for
    the evolution-target shortlist).
    `exclude_keys` drops specific `category/stem` keys (e.g. the seed page
    itself when called via `query_stem`).
    """
    if k <= 0:
        return []
    loaded = _scoped_load(scope)
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
               page_types: tuple[str, ...] | None = None, *,
               scope: EvidenceScope | None = None) -> list[PageHit]:
    """Top-k neighbors of an existing page identified by `category/stem`.

    The seed page is excluded from the result. Returns `[]` if the seed isn't
    indexed (e.g., empty page or built before the page was added).
    """
    if k <= 0 or (scope is not None and not scope.allows(key.rsplit('/', 1)[-1])):
        return []
    loaded = _scoped_load(scope)
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


def _scoped_load(scope):
    if scope is not None:
        scope.check_root()
        if not scope.stems:
            return None
    loaded = _load()
    if loaded is None or scope is None:
        return loaded
    arr, rows = loaded
    keep = [i for i, row in enumerate(rows) if scope.allows(row["stem"], row["page_type"])]
    return arr[keep], [rows[i] for i in keep]
