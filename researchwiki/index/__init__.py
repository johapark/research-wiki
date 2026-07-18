"""Indexing primitives — the shared substrate beneath `grade/` and `search/`.

Five focused modules:
  - embeddings   : bi-encoder model singleton + `embed_texts` (BAAI/bge-small)
  - types        : Document / SearchHit / SearchBackend ABC / SearchBackendUnavailable
  - pdf_chunks   : per-PDF Tantivy chunk index + chunk-embedding cache
  - pages_bm25   : Tantivy backend over wiki pages
  - pages_semantic: dense page-level embedding store

Consumers:
  - `grade/fidelity/`    — uses pdf_chunks + embeddings to score wiki pages
  - `search/__init__.py` — orchestration over pages_bm25
  - `search/hybrid.py`   — fuses pages_bm25 + pages_semantic via RRF
  - `agents/phases/...`  — uses pages_semantic for crosslink/evolution proposals
  - `tools/core.py`      — backs `researchwiki pdf-search` (ad-hoc PDF passage search)

Why this layering: the grader and the search backend grew independently and
each landed parallel implementations of "index over text". Semantic grading
and the semantic page index made the shared model loader explicit, and link
generation / memory evolution compounds the shared use. Pulling the
primitives into one package lets future "embed any
text" features pick the right index by name rather than guessing where the
loader lives.
"""

from __future__ import annotations
