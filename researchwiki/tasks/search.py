"""Query the wiki search index.

✅ Use when: you need to find wiki pages relevant to a topic or similar to
   a given page. Agents answering cross-paper questions should `--json`
   and feed the hits into the answer.
❌ Don't use: to list every page in the wiki (use `index.md` or `status`).
   Agent ingest updates indexes incrementally; reindex first only after manual
   bulk mutations or an ingest warning that the incremental update failed.

Two modes:
  - `researchwiki search "keywords"` — query.
  - `researchwiki search --like <category/stem>` — See-Also.

Retrieval mode (`--mode`):
  - `hybrid` (default when both indexes exist) — RRF fusion of BM25 + semantic.
  - `bm25`   — Tantivy keyword search only.
  - `semantic` — bi-encoder cosine similarity only (semantic page index).

Add `--see-also` to a keyword search to append 2-3 related pages per hit.
Add `--json` to get machine-parseable output instead of prose (agent-friendly).

Exit codes: 0 = hits returned; 1 = no hits (or `--like` stem not indexed);
2 = index not built (run `researchwiki reindex`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from ..index import pages_semantic as semantic_pages
from ..search import SearchBackendUnavailable, SearchHit, get_default_backend
from ..search import hybrid as hybrid_mod


def _fmt_hit(h: SearchHit, rank: int) -> str:
    title = (h.title or "").strip()
    if len(title) > 85:
        title = title[:82] + "…"
    line1 = f"{rank:2d}. {h.score:5.2f}  [[{h.key}]]  ({h.page_type})"
    line2 = f"      {title}"
    snippet = (h.snippet or "").strip()
    if snippet:
        if len(snippet) > 200:
            snippet = snippet[:197] + "…"
        return f"{line1}\n{line2}\n      › {snippet}"
    return f"{line1}\n{line2}"


def _fmt_hybrid(h: hybrid_mod.HybridHit, rank: int) -> str:
    """Hybrid hits show per-ranker rank inline so the user can see *why* a
    result fused well (matched on both signals) vs. one-sided (BM25 keyword
    hit but unrelated semantically, or vice versa)."""
    title = (h.title or "").strip()
    if len(title) > 85:
        title = title[:82] + "…"
    bm = f"BM25 #{h.bm25_rank}" if h.bm25_rank else "BM25 —"
    sem = f"sem #{h.semantic_rank} ({h.semantic_score:.2f})" if h.semantic_rank else "sem —"
    line1 = f"{rank:2d}. rrf={h.rrf_score:.4f}  [{bm}, {sem}]  [[{h.key}]]  ({h.page_type})"
    line2 = f"      {title}"
    snippet = (h.snippet or "").strip()
    if snippet:
        if len(snippet) > 200:
            snippet = snippet[:197] + "…"
        return f"{line1}\n{line2}\n      › {snippet}"
    return f"{line1}\n{line2}"


def _hit_as_dict(h: SearchHit) -> dict:
    d = dataclasses.asdict(h)
    d["key"] = h.key
    return d


def _hybrid_as_dict(h: hybrid_mod.HybridHit) -> dict:
    return {
        "key": h.key,
        "stem": h.stem,
        "category": h.category,
        "page_type": h.page_type,
        "title": h.title,
        "score": h.rrf_score,         # alias so callers expecting `score` work
        "rrf_score": h.rrf_score,
        "bm25_rank": h.bm25_rank,
        "bm25_score": h.bm25_score,
        "semantic_rank": h.semantic_rank,
        "semantic_score": h.semantic_score,
        "snippet": h.snippet,
    }


def _resolve_mode(requested: str) -> str:
    """`auto` → pick the best mode the environment supports.

    Hybrid requires both indexes; semantic requires the page index; BM25 is
    the universal fallback. We also fall through if `requested` is set but
    the required index is missing — better to return *something* than to
    error out before the user hits the index issue downstream.
    """
    if requested == "auto":
        if hybrid_mod.is_hybrid_available():
            return "hybrid"
        if semantic_pages.index_exists():
            return "semantic"
        return "bm25"
    if requested == "hybrid" and not hybrid_mod.is_hybrid_available():
        return "bm25" if not semantic_pages.index_exists() else "semantic"
    if requested == "semantic" and not semantic_pages.index_exists():
        return "bm25"
    return requested


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki search",
        description="Keyword / semantic / hybrid search and See-Also over the wiki.",
    )
    parser.add_argument("query", nargs="?", default=None,
                        help="Search terms (e.g. \"CRISPR off-target\"). "
                             "Tantivy syntax supported in --mode bm25/hybrid.")
    parser.add_argument("--like", metavar="CATEGORY/STEM", default=None,
                        help="Find documents similar to this wiki page (See-Also).")
    parser.add_argument("--mode", choices=["auto", "hybrid", "bm25", "semantic"],
                        default="auto",
                        help="Retrieval mode. `auto` (default) picks hybrid when "
                             "both indexes exist, else falls back gracefully.")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max results to return (default: 10)")
    parser.add_argument("--see-also", action="store_true",
                        help="For each keyword hit, also show 2 related pages.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a JSON array of hits instead of formatted prose.")
    args = parser.parse_args(argv)

    if (args.query is None) == (args.like is None):
        parser.error("specify exactly one of: positional query, or --like CATEGORY/STEM")

    mode = _resolve_mode(args.mode)

    if mode == "hybrid":
        return _run_hybrid(args)
    if mode == "semantic":
        return _run_semantic(args)
    return _run_bm25(args)


# ---------- BM25 path (legacy default; still used as fallback) ----------

def _run_bm25(args) -> int:
    backend = get_default_backend()
    try:
        if args.like:
            hits = backend.more_like(args.like, limit=args.limit)
            return _emit_hits(hits, args, header=f"# See-Also for [[{args.like}]] (BM25)")

        hits = backend.query(args.query, limit=args.limit)
        return _emit_hits(hits, args, header=f"# Search results for: {args.query!r}  (BM25; {len(hits)} hit{'s' if len(hits) != 1 else ''})",
                          extras_for=lambda h: backend.more_like(h.key, limit=2))
    except SearchBackendUnavailable as e:
        if args.as_json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"{e}", file=sys.stderr)
        return 2


def _emit_hits(hits, args, *, header: str, extras_for=None) -> int:
    if args.as_json:
        out = []
        for h in hits:
            d = _hit_as_dict(h)
            if args.see_also and extras_for is not None:
                d["see_also"] = [_hit_as_dict(r) for r in extras_for(h)]
            out.append(d)
        print(json.dumps(out, indent=2))
        return 0 if hits else 1
    if not hits:
        print("No hits.", file=sys.stderr)
        return 1
    print(header)
    print()
    for i, h in enumerate(hits, 1):
        print(_fmt_hit(h, i))
        if args.see_also and extras_for is not None:
            for r in extras_for(h):
                print(f"         → [[{r.key}]]  {(r.title or '')[:70]}")
        print()
    return 0


# ---------- semantic-only path ----------

def _run_semantic(args) -> int:
    if args.like:
        hits = semantic_pages.query_stem(args.like, k=args.limit)
        title = f"# See-Also for [[{args.like}]] (semantic)"
    else:
        hits = semantic_pages.query_text(args.query, k=args.limit)
        title = f"# Search results for: {args.query!r}  (semantic; {len(hits)} hit{'s' if len(hits) != 1 else ''})"

    # Adapt PageHit → SearchHit so the existing formatter works.
    sh = [SearchHit(stem=h.stem, category=h.category, page_type=h.page_type,
                    title=h.title, score=h.score, snippet="")
          for h in hits]
    return _emit_hits(sh, args, header=title, extras_for=None)


# ---------- hybrid (RRF) path ----------

def _run_hybrid(args) -> int:
    if args.like:
        hits = hybrid_mod.hybrid_more_like(args.like, limit=args.limit)
        header = f"# See-Also for [[{args.like}]] (hybrid RRF)"
        extras_for = lambda h: hybrid_mod.hybrid_more_like(h.key, limit=2)
    else:
        hits = hybrid_mod.hybrid_query(args.query, limit=args.limit)
        header = f"# Search results for: {args.query!r}  (hybrid RRF; {len(hits)} hit{'s' if len(hits) != 1 else ''})"
        extras_for = lambda h: hybrid_mod.hybrid_more_like(h.key, limit=2)

    if args.as_json:
        out = []
        for h in hits:
            d = _hybrid_as_dict(h)
            if args.see_also:
                d["see_also"] = [_hybrid_as_dict(r) for r in extras_for(h)]
            out.append(d)
        print(json.dumps(out, indent=2))
        return 0 if hits else 1

    if not hits:
        print("No hits.", file=sys.stderr)
        return 1

    print(header)
    print()
    for i, h in enumerate(hits, 1):
        print(_fmt_hybrid(h, i))
        if args.see_also:
            for r in extras_for(h):
                print(f"         → [[{r.key}]]  {(r.title or '')[:70]}")
        print()
    return 0
