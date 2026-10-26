"""Optional text-query encoders, isolated from the production claim grader."""

from __future__ import annotations

from ..benchmark.fixture import RetrievalFixture
from ..benchmark.retrieval import retrieve_pages
from ..benchmark.retrieval_embeddings import configure_model, _page_inputs
from ..index.pages_bm25 import TantivySearchBackend
from ..index.types import SearchBackendUnavailable, SearchHit


def query_with_model(query: str, model: str, *, mode: str = "auto", limit: int = 10,
                     max_seq_length: int = 2048) -> list[SearchHit]:
    """Use the same model-aware retrieval and cache as the comparison harness."""
    if limit <= 0 or not query.strip():
        return []
    configure_model(model, max_seq_length=max_seq_length)
    if mode == "auto":
        try:
            TantivySearchBackend().query("a", limit=1)
        except SearchBackendUnavailable:
            mode = "semantic"
        else:
            mode = "hybrid"
    fixture = RetrievalFixture("live-query", "pages", query, limit,
                               expected_claims=[], expected_pages=[], must_not_appear=[])
    ranked = retrieve_pages(fixture, model, mode)
    rows, _ = _page_inputs()
    by_key = {row["key"]: row for row in rows}
    return [SearchHit(stem=by_key[hit.paper_stem]["stem"],
                      category=by_key[hit.paper_stem]["category"],
                      page_type=by_key[hit.paper_stem]["page_type"],
                      title=by_key[hit.paper_stem]["title"], score=hit.score, snippet="")
            for hit in ranked if hit.paper_stem in by_key]
