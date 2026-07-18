"""Tantivy-backed `SearchBackend` implementation.

Tantivy is a Rust full-text engine (the same engine Obsidian uses for its
search plugin). We expose a thin wrapper that handles:

- Schema construction with per-field boosts.
- Index build/add (writer lifecycle, commit, reload).
- Keyword search with BM25 ranking + highlighted snippets.
- MoreLikeThis via "pass the reference text as a query" — tantivy-py does not
  expose MLT directly, but the equivalent is: extract the reference text and
  use it as the query. BM25's IDF weighting naturally promotes discriminative
  terms, which is exactly what MLT wants.
"""

from __future__ import annotations

from pathlib import Path

import tantivy

from ..paths import search_index_dir
from .types import Document, SearchBackend, SearchBackendUnavailable, SearchHit


# Default retrieval fields, with priorities implicit in the field list we pass
# to parse_query. Tantivy's Python bindings don't expose BM25 field boosts
# directly, so we emulate boosts by repeating a field's content across
# pseudo-boost fields at index time (title_boost contains the title again).
# Simpler: just let BM25 do its thing across (title, summary, body, authors)
# and rely on shorter fields getting naturally higher scores.
QUERY_FIELDS = ["title", "keywords", "summary", "body", "authors"]


def _build_schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    # Keyword fields (exact match only, no tokenization):
    sb.add_text_field("stem", stored=True, tokenizer_name="raw")
    sb.add_text_field("category", stored=True, tokenizer_name="raw")
    sb.add_text_field("page_type", stored=True, tokenizer_name="raw")
    sb.add_text_field("key", stored=True, tokenizer_name="raw")     # category/stem
    # Searchable text fields (default English analyzer):
    sb.add_text_field("title", stored=True, index_option="position")
    sb.add_text_field("authors", stored=True, index_option="position")
    sb.add_text_field("summary", stored=True, index_option="position")
    sb.add_text_field("keywords", stored=True, index_option="position")
    sb.add_text_field("body", stored=False, index_option="position")
    # Numeric:
    sb.add_unsigned_field("year", stored=True, indexed=True)
    return sb.build()


def _doc_to_hit(searcher: tantivy.Searcher, addr, score: float,
                snippet_gen: tantivy.SnippetGenerator | None) -> SearchHit:
    doc = searcher.doc(addr)
    d = doc.to_dict()
    snippet = ""
    if snippet_gen is not None:
        try:
            snippet = snippet_gen.snippet_from_doc(doc).fragment()
        except Exception:
            snippet = ""
    if not snippet:
        # fall back to first 160 chars of the summary (stored) or title
        summ = (d.get("summary") or [""])[0]
        snippet = (summ or (d.get("title") or [""])[0])[:160].replace("\n", " ")
    return SearchHit(
        stem=(d.get("stem") or [""])[0],
        category=(d.get("category") or [""])[0],
        page_type=(d.get("page_type") or ["paper"])[0],
        title=(d.get("title") or [""])[0],
        score=float(score),
        snippet=snippet,
    )


class TantivySearchBackend(SearchBackend):
    def __init__(self, path: Path | None = None):
        self._path = path or search_index_dir()
        self._schema = _build_schema()
        self._index: tantivy.Index | None = None

    @property
    def name(self) -> str:
        return "tantivy"

    # ---------- index lifecycle ----------

    def exists(self) -> bool:
        if not self._path.exists():
            return False
        # Tantivy marks a ready index with .managed.json.
        return (self._path / ".managed.json").exists()

    def _open(self) -> tantivy.Index:
        """Open an existing on-disk index for reading."""
        if self._index is not None:
            return self._index
        if not self.exists():
            raise SearchBackendUnavailable(
                f"No index at {self._path}. Run `researchwiki reindex` first."
            )
        self._index = tantivy.Index.open(str(self._path))
        self._index.reload()
        return self._index

    def _open_or_create(self) -> tantivy.Index:
        """Open existing or create a fresh empty index on disk."""
        self._path.mkdir(parents=True, exist_ok=True)
        if self.exists():
            self._index = tantivy.Index.open(str(self._path))
        else:
            self._index = tantivy.Index(self._schema, path=str(self._path))
        return self._index

    # ---------- writes ----------

    def build(self, docs: list[Document]) -> None:
        """Rebuild the index from scratch."""
        # Clear any existing index by wiping the directory.
        if self._path.exists():
            for p in sorted(self._path.glob("*")):
                try:
                    p.unlink()
                except IsADirectoryError:
                    import shutil
                    shutil.rmtree(p)
        self._index = None
        idx = self._open_or_create()
        writer = idx.writer(heap_size=30_000_000)
        for d in docs:
            writer.add_document(_to_tantivy_doc(d))
        writer.commit()
        writer.wait_merging_threads()
        idx.reload()

    def add(self, doc: Document) -> None:
        idx = self._open_or_create()
        writer = idx.writer(heap_size=15_000_000)
        writer.add_document(_to_tantivy_doc(doc))
        writer.commit()
        writer.wait_merging_threads()
        idx.reload()

    # ---------- reads ----------

    def query(self, q: str, limit: int = 10) -> list[SearchHit]:
        idx = self._open()
        searcher = idx.searcher()
        parsed = idx.parse_query(q, QUERY_FIELDS)
        res = searcher.search(parsed, limit=limit)
        snippet_gen = tantivy.SnippetGenerator.create(searcher, parsed, self._schema, "body")
        return [_doc_to_hit(searcher, addr, s, snippet_gen) for s, addr in res.hits]

    def more_like(self, key: str, limit: int = 10) -> list[SearchHit]:
        """Find documents similar to the one at `category/stem`."""
        idx = self._open()
        searcher = idx.searcher()
        # Look up the seed document by its `key` (category/stem).
        key_q = idx.parse_query(_escape_for_raw_field(key), ["key"])
        seed = searcher.search(key_q, limit=1)
        if not seed.hits:
            return []
        seed_doc = searcher.doc(seed.hits[0][1])
        dd = seed_doc.to_dict()
        title = (dd.get("title") or [""])[0]
        summary = (dd.get("summary") or [""])[0]
        # Use title + summary as the similarity query. Avoid `body` as seed
        # because very long bodies swamp IDF.
        seed_text = f"{title}\n{summary}"
        hits = self.more_like_text(seed_text, limit=limit + 1)
        return [h for h in hits if h.key != key][:limit]

    def more_like_text(
        self,
        text: str,
        limit: int = 10,
        page_type: str | None = None,
    ) -> list[SearchHit]:
        idx = self._open()
        searcher = idx.searcher()
        # Use parse_query_lenient so punctuation in the seed text doesn't blow up.
        # The lenient parser returns (query, errors); we ignore errors.
        try:
            parsed, _errors = tantivy.parse_query_lenient(
                _sanitize_query(text), idx.schema, QUERY_FIELDS
            )
        except Exception:
            # Fallback: try the strict parser on a heavily-sanitized text
            parsed = idx.parse_query(_sanitize_query(text, aggressive=True), QUERY_FIELDS)
        res = searcher.search(parsed, limit=limit * 2 if page_type else limit)
        snippet_gen = tantivy.SnippetGenerator.create(searcher, parsed, self._schema, "body")
        hits = [_doc_to_hit(searcher, addr, s, snippet_gen) for s, addr in res.hits]
        if page_type is not None:
            hits = [h for h in hits if h.page_type == page_type]
        return hits[:limit]


# ---------- helpers ----------

def _to_tantivy_doc(d: Document) -> tantivy.Document:
    td = tantivy.Document()
    td.add_text("stem", d.stem)
    td.add_text("category", d.category)
    td.add_text("page_type", d.page_type)
    td.add_text("key", d.key)
    td.add_text("title", d.title or "")
    td.add_text("authors", d.authors or "")
    td.add_text("summary", d.summary or "")
    td.add_text("keywords", d.keywords or "")
    td.add_text("body", d.body or "")
    if d.year is not None:
        try:
            td.add_unsigned("year", int(d.year))
        except (ValueError, TypeError):
            pass
    return td


def _escape_for_raw_field(s: str) -> str:
    """Escape characters that would confuse the query parser when searching
    a keyword ('raw' tokenizer) field."""
    # For raw-tokenizer fields tantivy needs the value quoted as a phrase.
    return f'"{s}"'


# Characters that Tantivy's query parser treats specially. We strip them when
# building a similarity query so that the seed text can't accidentally parse
# as a boolean expression.
_QUERY_METACHARS = str.maketrans({c: " " for c in r'+-!(){}[]^"~*?:\/'})


def _sanitize_query(text: str, aggressive: bool = False) -> str:
    t = text.translate(_QUERY_METACHARS)
    # Collapse repeated whitespace.
    t = " ".join(t.split())
    if aggressive:
        # Remove any residual non-alphanumeric runs.
        import re
        t = re.sub(r"[^A-Za-z0-9 \-]+", " ", t)
        t = " ".join(t.split())
    # Cap length — seed abstracts run long and we don't need every token.
    return t[:2000]
