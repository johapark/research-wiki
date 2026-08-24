"""Rebuild the full-text and semantic search indexes from the current `wiki/` state.

✅ Use when: pages were deleted, manually/bulk edited outside ingest, or an
   ingest reported an incremental-index warning. Successful agent ingests
   upsert only the pages they changed, so routine ingest needs no full rebuild.
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
            # Thin-vector warning. Reported here rather than only in `lint`
            # because this is where the text is actually assembled, so the
            # warning cannot disagree with what got embedded. A page that embeds
            # only its title is invisible to see-also, ingest cross-linking,
            # `evolve`'s KNN and `candidates synthesis` — and stays invisible
            # silently, which is how 41 pages went unnoticed on 2026-08-06.
            thin = result.get("thin") or []
            if thin:
                print()
                print(f"⚠ {len(thin)} page(s) embed too little text to retrieve on:")
                for e in thin[:15]:
                    print(f"  - {e['key']}  ({e['page_type']}) — {e['reason']}")
                if len(thin) > 15:
                    print(f"  - ... ({len(thin) - 15} more)")
                print("  Check the page's H2 names against `index.pages_semantic."
                      "_INDEX_SECTIONS` for its type.")

    return 0
