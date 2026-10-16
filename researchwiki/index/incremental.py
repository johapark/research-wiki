"""Page maintenance shares corpus → index/cache → target lock ordering.

Read canonical pages only after taking the corpus guard. Recovery can maintain
keyword search without constructing an embedding model. Page embeddings are
computed outside the corpus guard and published under it (see
`_refresh_page_vectors`).
"""

from pathlib import Path
import time

from ..fsatomic import exclusive_lock
from ..mutation import corpus_lock
from ..paths import search_index_dir, wiki_dir
from ..search import build_documents_from_wiki, document_from_page, get_default_backend
from ..wiki import read_page, read_pages
from . import pages_semantic


def update_page_indexes(paths: list[Path | str], *, semantic: bool = True,
                        mark_semantic_stale: bool = True) -> dict:
    """Upsert current pages; absent paths require a full deletion reconciliation.

    With `semantic=False`, the page vectors are marked stale unless the caller
    has already reconciled them itself (`mark_semantic_stale=False`).
    """
    return _maintain(paths, semantic=semantic, mark_semantic_stale=mark_semantic_stale)


def rebuild_page_indexes(*, semantic: bool = True) -> dict:
    """Rebuild both projections from one locked snapshot, including an empty wiki."""
    return _maintain(None, semantic=semantic)


def _maintain(paths, *, semantic, mark_semantic_stale=True):
    with corpus_lock(), exclusive_lock(search_index_dir()):
        full = paths is None
        scope: list[Path] = []
        if full:
            pages = read_pages()
        else:
            root = wiki_dir().resolve()
            scope = sorted({Path(p).resolve() for p in paths if Path(p).suffix.lower() == ".md"
                            and root in Path(p).resolve().parents})
            full = any(not path.exists() for path in scope)
            pages = read_pages() if full else [page for path in scope
                                               if (page := read_page(path)) is not None]
            if not pages and not full:
                return {"n_pages": 0, "documents": [], "bm25_mode": "skipped",
                        "semantic_mode": "skipped", "semantic_embedded": None, "semantic": None}
        backend = get_default_backend()
        bm25_started = time.monotonic()
        if full or not backend.exists():
            docs = build_documents_from_wiki()
            backend.build(docs)
            bm25_mode = "rebuilt"
        else:
            docs = [document_from_page(page) for page in pages]
            for doc in docs:
                backend.add(doc)
            bm25_mode = "incremental"
        bm25_seconds = time.monotonic() - bm25_started
        if not semantic and mark_semantic_stale:
            pages_semantic.mark_stale()

    semantic_started = time.monotonic()
    sem = _refresh_page_vectors(pages, scope, full) if semantic else None
    return {
        "n_pages": len(pages), "documents": docs,
        "bm25_mode": bm25_mode,
        "semantic_mode": (sem or {}).get("mode", "rebuilt" if sem else "skipped"),
        "semantic_embedded": (sem or {}).get("n_embedded"),
        "semantic": sem,
        "bm25_seconds": bm25_seconds,
        "semantic_seconds": time.monotonic() - semantic_started,
    }


def _refresh_page_vectors(pages, scope: list[Path], full: bool) -> dict | None:
    """Embed without the corpus guard, then publish under it.

    Holding the guard through embedding blocked every concurrent promote,
    `db rebuild` and grade persistence for the whole run — minutes on a full
    `reindex`. Vectors are prepared from this snapshot first; the pages are
    then re-read under the guard, and `prepare_vectors` keys each vector by the
    text it encodes, so a page edited in between is embedded again there
    rather than published with a vector for its old text.
    """
    prepared = pages_semantic.prepare_vectors(pages, full=full)
    with corpus_lock():
        if full:
            pages = read_pages()
        else:
            vanished = {path.stem for path in scope if not path.exists()}
            if vanished:
                # Deleted between the two windows: drop their rows without a model.
                pages_semantic.drop_outdated([], vanished)
            pages = [page for path in scope if (page := read_page(path)) is not None]
        try:
            sem = (pages_semantic.build_index(pages, prepared=prepared) if full
                   else pages_semantic.upsert_pages(pages, prepared=prepared))
        except Exception:
            pages_semantic.mark_stale()
            raise
        if sem is None:
            pages_semantic.mark_stale()
    return sem
