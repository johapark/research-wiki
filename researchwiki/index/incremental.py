"""Incremental page-index maintenance after wiki mutations.

Markdown remains canonical. Existing indexes receive page-level upserts; a
missing or corrupt index falls back to the same full builder used by
``researchwiki reindex``. One cross-process lock covers BM25 and semantic
updates so parallel ingests cannot clobber the aligned semantic files.
"""

from __future__ import annotations

from pathlib import Path

from ..fsatomic import exclusive_lock
from ..paths import search_index_dir
from ..search import (
    build_documents_from_wiki,
    document_from_page,
    get_default_backend,
)
from ..wiki import read_page, read_pages
from . import pages_semantic


def update_page_indexes(paths: list[Path | str]) -> dict:
    """Upsert parseable pages at ``paths`` into both page indexes.

    Returns modes/counts for logging. Non-page files are ignored. Failures are
    intentionally allowed to propagate to the ingest caller, which records a
    warning and points at the full ``reindex`` recovery path.
    """
    pages = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        resolved = path.resolve()
        if resolved in seen or not path.exists() or path.suffix.lower() != ".md":
            continue
        seen.add(resolved)
        page = read_page(path)
        if page is not None:
            pages.append(page)
    if not pages:
        return {"n_pages": 0, "bm25_mode": "skipped", "semantic_mode": "skipped"}

    with exclusive_lock(search_index_dir()):
        backend = get_default_backend()
        if backend.exists():
            for page in pages:
                backend.add(document_from_page(page))
            bm25_mode = "incremental"
        else:
            backend.build(build_documents_from_wiki())
            bm25_mode = "rebuilt"

        if pages_semantic.index_exists():
            sem = pages_semantic.upsert_pages(pages)
        else:
            sem = pages_semantic.build_index(read_pages())
            if sem is not None:
                sem["mode"] = "rebuilt"

    return {
        "n_pages": len(pages),
        "bm25_mode": bm25_mode,
        "semantic_mode": (sem or {}).get("mode", "skipped"),
        "semantic_embedded": (sem or {}).get("n_embedded"),
    }
