"""Rebuild the full-text and semantic search indexes from the current `wiki/` state.

✅ Use when: a wiki page was added, deleted, or substantially edited and
   you want `search` / See-Also / category auto-suggest / Phase-B candidate
   selection to see the change. Fast enough to run every time.
❌ Don't use: as a fix for a bad search result — the ranking issue is
   probably in the query, not staleness.

Two indexes are rebuilt in lockstep:
  - Tantivy BM25 at `.tantivy-index/` — backs `search`, See-Also, category
    auto-suggest in `researchwiki ingest`.
  - Page-level dense embeddings at `.semantic-cache/` — backs Phase-B link
    generation, memory evolution candidate selection, and hybrid retrieval.

Pass `--no-semantic` to skip the embedding pass (e.g., when the bi-encoder
isn't available or you only need keyword search).

Exit code: 0 on success; 2 if either index directory can't be written.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from ..log import log
from ..index import pages_semantic as semantic_pages
from ..search import build_documents_from_wiki, get_default_backend
from ..wiki import read_pages


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki reindex",
        description="Rebuild Tantivy + semantic page indexes from `wiki/`.",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip the page-level semantic embedding pass.",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    docs = build_documents_from_wiki()
    if not docs:
        log("no wiki pages found; nothing to index", tag="reindex")
        return 0

    backend = get_default_backend()
    backend.build(docs)
    tantivy_dt = time.time() - t0

    counts = Counter(d.page_type for d in docs)
    breakdown = ", ".join(f"{n} {t}" for t, n in sorted(counts.items()))
    print(f"Indexed {len(docs)} pages ({breakdown}) in {tantivy_dt:.2f}s "
          f"→ .tantivy-index/")

    semantic_summary = ""
    if not args.no_semantic:
        t1 = time.time()
        # Re-read to get Page objects with frontmatter — Document objects
        # drop fields we want for the embedding text (tags, keywords).
        pages = read_pages()
        result = semantic_pages.build_index(pages)
        sem_dt = time.time() - t1
        if result is None:
            semantic_summary = (
                "Semantic index skipped (model unavailable; install "
                "sentence-transformers or pass --no-semantic)."
            )
            print(semantic_summary)
        else:
            semantic_summary = (
                f"Embedded {result['n_pages']} pages "
                f"(dim={result['dim']}) in {sem_dt:.2f}s → .semantic-cache/"
            )
            print(semantic_summary)

    return 0
