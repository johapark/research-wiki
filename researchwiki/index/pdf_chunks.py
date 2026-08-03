"""Per-PDF Tantivy chunk index.

Each paper gets its own Tantivy index under `.grade-cache/{stem}/`, populated
once from the PDF text and reused on every grade call. We index chunks
(~250-word sliding windows with overlap) so retrieval can return both a score
and the supporting passage — the passage is what the semantic scorer
consumes for embedding-based grounding, and what a human can eyeball to
verify the BM25 signal. Pre-computed chunk embeddings live next to the
Tantivy index so per-claim scoring is one query embedding + a dot product.

Single-PDF indexes (vs. one big index with a stem field) keep BM25 statistics
local: a claim's match against the parent paper isn't penalized by frequent
terms in unrelated papers.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tantivy

from ..paths import grade_cache_dir, resolve_pdf
from ..pdf.text import extract_pdf


CHUNK_WORDS = 250
OVERLAP_WORDS = 60
MAX_PDF_PAGES = 80
TOPK_DEFAULT = 5
EMBEDDINGS_FILENAME = "embeddings.npy"
EMBEDDINGS_META_FILENAME = "embeddings_meta.json"


@dataclass
class Chunk:
    chunk_id: int
    text: str


@dataclass
class RetrievedChunk:
    chunk_id: int
    score: float
    text: str


def _chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    """Sliding-window chunker. Whitespace-split; chunks overlap by `overlap` words.

    For dense scientific text, ~250 words ≈ 350 tokens — well under typical
    sentence-encoder context limits (512), with enough surrounding context
    that a single claim's supporting evidence usually fits in one chunk.
    """
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks: list[Chunk] = []
    cid = 0
    i = 0
    while i < len(words):
        window = words[i : i + chunk_words]
        if len(window) < 30 and chunks:
            # Trailing tail too short to be meaningful; skip
            break
        chunks.append(Chunk(chunk_id=cid, text=" ".join(window)))
        cid += 1
        if i + chunk_words >= len(words):
            break
        i += step
    return chunks


def _build_schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    sb.add_unsigned_field("chunk_id", stored=True, indexed=True)
    sb.add_text_field("text", stored=True, index_option="position")
    return sb.build()


def _index_path_for(stem: str) -> Path:
    return grade_cache_dir() / stem


def build_pdf_index(
    stem: str,
    force: bool = False,
    pdf_path: Path | str | None = None,
) -> Path:
    """Build (or reuse) the chunk index for a single paper.

    Returns the index directory. Idempotent: if the index already exists and
    `force` is False, returns immediately.

    Args:
      stem: identifier for the index directory (.grade-cache/{stem}/).
      force: rebuild even if the index already exists.
      pdf_path: explicit PDF path. When None, the PDF is looked up at
                `papers/{stem}.pdf` — the canonical post-ingest location.
                Pass an explicit path to grade a draft against a PDF that
                lives elsewhere (e.g. still in inbox/).
    """
    idx_dir = _index_path_for(stem)
    if idx_dir.exists() and not force:
        return idx_dir
    if idx_dir.exists() and force:
        shutil.rmtree(idx_dir)

    src = Path(pdf_path) if pdf_path is not None else resolve_pdf(stem)
    if not src.exists():
        raise FileNotFoundError(f"PDF not found: {src}")
    pdf_path = src
    text, _meta = extract_pdf(pdf_path, max_pages=MAX_PDF_PAGES)
    chunks = _chunk_text(text)
    if not chunks:
        raise RuntimeError(f"No chunks extracted from {pdf_path}; PDF may be unparseable.")

    idx_dir.mkdir(parents=True, exist_ok=True)
    schema = _build_schema()
    index = tantivy.Index(schema, path=str(idx_dir))
    writer = index.writer(heap_size=15_000_000)
    for ch in chunks:
        doc = tantivy.Document()
        doc.add_unsigned("chunk_id", ch.chunk_id)
        doc.add_text("text", ch.text)
        writer.add_document(doc)
    writer.commit()
    writer.wait_merging_threads()

    # Embedding cache built alongside the BM25 index. Failures are non-fatal:
    # if the embedding model isn't installed, BM25 retrieval still works and
    # the semantic scorer will report unavailable.
    _build_embeddings_cache(idx_dir, [c.text for c in chunks])
    return idx_dir


def _build_embeddings_cache(idx_dir: Path, chunk_texts: list[str]) -> None:
    """Embed the chunks and persist alongside the Tantivy index.

    Stores the embedding matrix (float32, L2-normalized) plus a sidecar
    metadata JSON with model name, dim, and chunk texts so the scorer can
    reload everything in one pass. Silent no-op if the embedding backend is
    unavailable — semantic.is_available() is the canonical probe.
    """
    from . import embeddings as semantic
    embs = semantic.embed_texts(chunk_texts)
    if embs is None or embs.size == 0:
        return
    np.save(idx_dir / EMBEDDINGS_FILENAME, embs)
    meta = {
        "model": semantic.DEFAULT_MODEL,
        "dim": int(embs.shape[1]),
        "n_chunks": int(embs.shape[0]),
        "chunk_texts": chunk_texts,
    }
    (idx_dir / EMBEDDINGS_META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")


def get_chunk_embeddings(stem: str) -> tuple[np.ndarray, list[str]] | None:
    """Return (embeddings, chunk_texts) for a paper. Builds on demand if
    the cache is missing. Returns None if the index can't be built or the
    embedding model is unavailable.
    """
    idx_dir = _index_path_for(stem)
    if not idx_dir.exists():
        try:
            build_pdf_index(stem)
        except Exception:
            return None
    emb_path = idx_dir / EMBEDDINGS_FILENAME
    meta_path = idx_dir / EMBEDDINGS_META_FILENAME
    if not emb_path.exists() or not meta_path.exists():
        # Index pre-dates the embedding cache; rebuild just the embeddings
        # in-place so callers don't have to drop the whole Tantivy index.
        chunk_texts = _read_cached_chunk_texts(idx_dir)
        if chunk_texts is None:
            return None
        _build_embeddings_cache(idx_dir, chunk_texts)
        if not emb_path.exists():
            return None
    embs = np.load(emb_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return embs, list(meta.get("chunk_texts") or [])


def _read_cached_chunk_texts(idx_dir: Path) -> list[str] | None:
    """Recover chunk texts from a Tantivy index that was built before the
    embedding cache existed. Walks all stored documents in chunk_id order.
    """
    try:
        index = tantivy.Index.open(str(idx_dir))
        index.reload()
        searcher = index.searcher()
        all_query = index.parse_query("*", ["text"])
        res = searcher.search(all_query, limit=10_000)
    except Exception:
        return None
    pairs = []
    for _score, addr in res.hits:
        d = searcher.doc(addr).to_dict()
        cid = int((d.get("chunk_id") or [0])[0])
        text = (d.get("text") or [""])[0]
        pairs.append((cid, text))
    pairs.sort(key=lambda p: p[0])
    return [t for _, t in pairs]


def query_pdf(stem: str, query: str, topk: int = TOPK_DEFAULT) -> list[RetrievedChunk]:
    """Run a BM25 query against the paper's chunk index. Returns top-k chunks."""
    idx_dir = _index_path_for(stem)
    if not idx_dir.exists():
        build_pdf_index(stem)
    index = tantivy.Index.open(str(idx_dir))
    index.reload()
    searcher = index.searcher()
    try:
        parsed, _ = index.parse_query_lenient(query, ["text"])
    except Exception:
        # Final fallback: strip non-alphanumeric characters and lowercase, so
        # reserved boolean operators (AND/OR/NOT) survive as ordinary terms
        # rather than tripping the strict parser. Best-effort: an unparseable
        # claim scores zero rather than aborting the grade phase.
        cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in query).lower()
        try:
            parsed = index.parse_query(cleaned, ["text"])
        except Exception:
            return []
    res = searcher.search(parsed, limit=topk)
    out: list[RetrievedChunk] = []
    for score, addr in res.hits:
        d = searcher.doc(addr).to_dict()
        out.append(
            RetrievedChunk(
                chunk_id=int((d.get("chunk_id") or [0])[0]),
                score=float(score),
                text=(d.get("text") or [""])[0],
            )
        )
    return out
