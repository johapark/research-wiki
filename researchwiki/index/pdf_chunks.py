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
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tantivy

from ..paths import grade_cache_dir, resolve_pdf
from ..fsatomic import exclusive_lock, file_sha256, write_json_atomic
from .bundle import load_bundle, save_bundle
from ..pdf.sections import section_for_offset, section_spans
from ..pdf.text import (
    PAGE_SEPARATOR,
    extract_pdf_page_texts,
    page_for_offset,
    page_offsets,
)


CHUNK_WORDS = 250
OVERLAP_WORDS = 60
MAX_PDF_PAGES = 80
TOPK_DEFAULT = 5
EMBEDDINGS_FILENAME = "embeddings.npy"
EMBEDDINGS_META_FILENAME = "embeddings_meta.json"
INDEX_META_FILENAME = "index_meta.json"

# Bump when the Tantivy schema or the chunking changes. `build_pdf_index`
# returns early when the index directory exists, so without a recorded version
# a schema change would silently leave every existing index in place, missing
# the new fields and reporting no error. Nothing else in `.grade-cache/`
# carries a format version — `embeddings_meta.json` records the model and dim,
# which is a different question.
#   1 — (implicit) chunk_id + text
#   2 — page_start / page_end / section provenance
CACHE_VERSION = 2

_WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    chunk_id: int
    text: str
    # Provenance, all optional: a PDF with no detectable headings yields no
    # spans, and `None` is the honest answer there. A wrong section label on
    # displayed evidence is worse than no label.
    page_start: int | None = None   # 1-based, as the renderer numbers pages
    page_end: int | None = None     # equal to page_start unless the window spans a break
    section: str | None = None      # PDF section name, e.g. 'results'

    def provenance(self) -> str:
        """`§results, p. 7` / `p. 7-8` / `''` — the display form.

        Note this is the *PDF's* section, which is a different axis from the
        `claims.section` column (that one names the wiki page's H2:
        key_contributions / results / limitations / methodology).
        """
        bits = []
        if self.section:
            bits.append(f"§{self.section}")
        if self.page_start:
            if self.page_end and self.page_end != self.page_start:
                bits.append(f"pp. {self.page_start}-{self.page_end}")
            else:
                bits.append(f"p. {self.page_start}")
        return ", ".join(bits)


@dataclass
class RetrievedChunk:
    chunk_id: int
    score: float
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None

    def provenance(self) -> str:
        """Display form, e.g. `§results, p. 7`. Empty when unlabelled."""
        return Chunk(
            chunk_id=self.chunk_id, text=self.text,
            page_start=self.page_start, page_end=self.page_end,
            section=self.section,
        ).provenance()


def _chunk_text(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap: int = OVERLAP_WORDS,
    *,
    page_starts: list[int] | None = None,
    spans: list[tuple[int, int, str]] | None = None,
) -> list[Chunk]:
    """Sliding-window chunker. Whitespace-split; chunks overlap by `overlap` words.

    For dense scientific text, ~250 words ≈ 350 tokens — well under typical
    sentence-encoder context limits (512), with enough surrounding context
    that a single claim's supporting evidence usually fits in one chunk.

    `page_starts` (from `pdf.text.page_offsets`) and `spans` (from
    `pdf.sections.section_spans`) are both optional. Supplied, each chunk is
    labelled with the page range it covers and the section its *start* falls
    in; omitted, the chunk text is unchanged and the labels are None. Both index
    into the same string this is chunking, which is why `extract_pdf_page_texts`
    guarantees its join reproduces `extract_pdf` exactly.

    Words are located with a regex rather than `str.split()` so each window
    knows its character span; the emitted text is still the joined words, so
    chunk text is byte-identical to what the previous implementation produced.
    """
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return []
    step = max(1, chunk_words - overlap)
    chunks: list[Chunk] = []
    cid = 0
    i = 0
    while i < len(matches):
        window = matches[i : i + chunk_words]
        if len(window) < 30 and chunks:
            # Trailing tail too short to be meaningful; skip
            break
        start_off, end_off = window[0].start(), window[-1].end()
        page_start = page_end = None
        if page_starts:
            page_start = page_for_offset(page_starts, start_off, len(text))
            page_end = page_for_offset(page_starts, end_off - 1, len(text)) or page_start
        chunks.append(Chunk(
            chunk_id=cid,
            text=" ".join(m.group() for m in window),
            page_start=page_start,
            page_end=page_end,
            # The start's section, not the end's: a window straddling a
            # heading belongs to the section it opens in, which is where its
            # first and most-weighted sentences are.
            section=section_for_offset(spans, start_off) if spans else None,
        ))
        cid += 1
        if i + chunk_words >= len(matches):
            break
        i += step
    return chunks


def _build_schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    sb.add_unsigned_field("chunk_id", stored=True, indexed=True)
    sb.add_text_field("text", stored=True, index_option="position")
    # Provenance: stored for display, not indexed — nothing queries by page or
    # section, and indexing them would change BM25 term statistics.
    sb.add_unsigned_field("page_start", stored=True, indexed=False)
    sb.add_unsigned_field("page_end", stored=True, indexed=False)
    sb.add_text_field("section", stored=True, index_option="basic")
    return sb.build()


def _index_path_for(stem: str) -> Path:
    return grade_cache_dir() / stem


def _read_index_version(idx_dir: Path) -> int:
    """Recorded cache version, 1 for an index written before versioning."""
    try:
        meta = json.loads((idx_dir / INDEX_META_FILENAME).read_text(encoding="utf-8"))
        return int(meta.get("cache_version", 1))
    except Exception:
        return 1


def _index_metadata(idx_dir: Path) -> dict:
    try:
        metadata = json.loads((idx_dir / INDEX_META_FILENAME).read_text(encoding="utf-8"))
        return metadata if isinstance(metadata, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_index_meta(idx_dir: Path, source: Path | None = None, digest: str | None = None) -> None:
    write_json_atomic(idx_dir / INDEX_META_FILENAME, {
        "cache_version": CACHE_VERSION, "source_path": str(source.resolve()) if source else None,
        "source_pdf_sha256": digest,
    })


def _source(stem: str) -> Path:
    try:
        return resolve_pdf(stem)
    except FileNotFoundError:
        source = _index_metadata(_index_path_for(stem)).get("source_path")
        if isinstance(source, str) and source:
            return Path(source)
        raise


def _locked(function):
    from functools import wraps
    @wraps(function)
    def call(stem, *args, **kwargs):
        with exclusive_lock(_index_path_for(stem)):
            return function(stem, *args, **kwargs)
    return call


@_locked
def build_pdf_index(
    stem: str,
    force: bool = False,
    pdf_path: Path | str | None = None,
) -> Path:
    """Build (or reuse) the chunk index for a single paper.

    Returns the index directory. Reuses a complete index only when its cache
    version and source PDF hash match; otherwise stages a fresh index.

    Args:
      stem: identifier for the index directory (.grade-cache/{stem}/).
      force: rebuild even if the index already exists.
      pdf_path: explicit PDF path. When None, the PDF is looked up at
                `papers/{stem}.pdf` — the canonical post-ingest location.
                Pass an explicit path to grade a draft against a PDF that
                lives elsewhere (e.g. still in inbox/).
    """
    target = _index_path_for(stem)
    src = Path(pdf_path) if pdf_path is not None else _source(stem)
    if not src.is_file():
        raise FileNotFoundError(f"PDF not found: {src}")
    digest = file_sha256(src)
    meta = _index_metadata(target)
    if (not force and meta.get("cache_version") == CACHE_VERSION
            and meta.get("source_pdf_sha256") == digest
            and (target / ".managed.json").exists()):
        return target
    pdf_path = src
    pages = extract_pdf_page_texts(pdf_path, max_pages=MAX_PDF_PAGES)
    text = PAGE_SEPARATOR.join(pages)
    chunks = _chunk_text(
        text, page_starts=page_offsets(pages), spans=section_spans(text)
    )
    if not chunks:
        raise RuntimeError(f"No chunks extracted from {pdf_path}; PDF may be unparseable.")

    target.parent.mkdir(parents=True, exist_ok=True)
    idx_dir = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=target.parent))
    try:
        schema = _build_schema()
        index = tantivy.Index(schema, path=str(idx_dir))
        writer = index.writer(heap_size=15_000_000)
        for ch in chunks:
            doc = tantivy.Document()
            doc.add_unsigned("chunk_id", ch.chunk_id)
            doc.add_text("text", ch.text)
            # Tantivy has no null: a page we couldn't place is simply absent from
            # the document, and `_stored` reads it back as None.
            if ch.page_start:
                doc.add_unsigned("page_start", ch.page_start)
            if ch.page_end:
                doc.add_unsigned("page_end", ch.page_end)
            if ch.section:
                doc.add_text("section", ch.section)
            writer.add_document(doc)
        writer.commit()
        writer.wait_merging_threads()
        del writer, index  # Close native handles before directory replacement.

        # Embedding cache built alongside the BM25 index. Failures are non-fatal:
        # if the embedding model isn't installed, BM25 retrieval still works and
        # the semantic scorer will report unavailable.
        _build_embeddings_cache(idx_dir, [c.text for c in chunks])
        if file_sha256(src) != digest:
            shutil.rmtree(idx_dir, ignore_errors=True)
            raise RuntimeError(f"PDF changed while indexing: {src}")
        _write_index_meta(idx_dir, src, digest)
        if target.exists():
            shutil.rmtree(target)
        idx_dir.replace(target)
        return target
    finally:
        shutil.rmtree(idx_dir, ignore_errors=True)


def _build_embeddings_cache(idx_dir: Path, chunk_texts: list[str]) -> None:
    """Embed the chunks and persist alongside the Tantivy index.

    Stores the embedding matrix and model/dimension/chunk-text metadata in
    one atomic NumPy bundle. Silent no-op if the embedding backend is
    unavailable — semantic.is_available() is the canonical probe.
    """
    from . import embeddings as semantic
    embs = semantic.embed_texts(chunk_texts)
    if embs is None or embs.size == 0:
        return
    save_bundle(idx_dir / "embeddings.npz", embs, {
        "model": semantic.DEFAULT_MODEL, "dim": int(embs.shape[1]),
        "rows": [{"text": text} for text in chunk_texts],
    })


@_locked
def get_chunk_embeddings(stem: str) -> tuple[np.ndarray, list[str]] | None:
    """Return (embeddings, chunk_texts) for a paper. Builds on demand if
    the cache is missing. Returns None if the index can't be built or the
    embedding model is unavailable.
    """
    from . import embeddings as semantic
    try:
        idx_dir = build_pdf_index(stem)
        loaded = load_bundle(idx_dir / "embeddings.npz", semantic.DEFAULT_MODEL, row_fields=("text",))
        if loaded is None:
            texts = _read_cached_chunk_texts(idx_dir)
            if texts is None:
                return None
            _build_embeddings_cache(idx_dir, texts)
            loaded = load_bundle(idx_dir / "embeddings.npz", semantic.DEFAULT_MODEL, row_fields=("text",))
        return (loaded[0], [r["text"] for r in loaded[1]["rows"]]) if loaded else None
    except (OSError, RuntimeError):
        return None


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


#: A `-` or `+` Tantivy would read as an operator: one that starts a term,
#: after start-of-text or whitespace. Interior hyphens (`multi-allelic`,
#: `vg-giraffe`) and minus signs inside numbers are left alone.
_BOOLEAN_PREFIX_RE = re.compile(r"(^|\s)[-+]+(?=\S)")
#: A lone `-`/`+` token, such as a Markdown bullet marker or a spaced dash.
_LONE_OPERATOR_RE = re.compile(r"(^|\s)[-+]+(?=\s|$)")


def _neutralize_operators(query: str) -> str:
    """Drop `-`/`+` that Tantivy's query parser would treat as operators.

    Callers pass claim prose, not search syntax. Markdown list items start
    with `- `, and `parse_query_lenient` reads that as MUST_NOT on the next
    term (`- The panel…` excludes every chunk containing "the"), which
    silently returns no hits. The synthesis grader then scored a correctly
    cited bullet as `weak`: 24 of 492 bullet claims on this corpus's
    synthesis and idea pages lost all retrieval this way.
    """
    query = _LONE_OPERATOR_RE.sub(r"\1", query)
    return _BOOLEAN_PREFIX_RE.sub(r"\1", query)


@_locked
def query_pdf(stem: str, query: str, topk: int = TOPK_DEFAULT) -> list[RetrievedChunk]:
    """Run a BM25 query against the paper's chunk index. Returns top-k chunks."""
    query = _neutralize_operators(query)
    idx_dir = build_pdf_index(stem)
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

        def _first(key, cast, document=d):
            vals = document.get(key)
            return cast(vals[0]) if vals else None

        out.append(
            RetrievedChunk(
                chunk_id=int((d.get("chunk_id") or [0])[0]),
                score=float(score),
                text=(d.get("text") or [""])[0],
                page_start=_first("page_start", int),
                page_end=_first("page_end", int),
                section=_first("section", str),
            )
        )
    return out
