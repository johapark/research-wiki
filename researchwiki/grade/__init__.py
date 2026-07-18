"""Coverage grader — quality scoring for wiki pages against their source PDFs.

The grader provides non-LLM oracles for two orthogonal quality axes:

  Fidelity (precision) — "does this wiki page accurately reflect the source
                          PDF?" Per-claim retrieval against the PDF and
                          deterministic numeric/negation integrity checks.
  Salience (recall)    — "does the page capture what the PDF flagged as
                          load-bearing?" A synthetic ContentFixture is built
                          from PDF anchors (abstract proxy, Results /
                          Discussion lead-ins, figure captions) and scored
                          via the benchmark-fixture scorer. See `salience.py`.

Both axes are answerable as a query rather than a re-read, which unblocks
tournament-based ingestion, proactive synthesis, and weak-page surfacing.

Fidelity signals:
  - BM25 retrieval over a per-PDF Tantivy chunk index (cheap; per-claim top-1
    BM25 score).
  - Bi-encoder semantic similarity via BAAI/bge-small-en-v1.5 (per-claim
    max cosine over BM25 top-10 retrieved chunks; chunk embeddings cached
    at index-build time).
  - Deterministic negation parity check (claim has negation, top chunks
    don't → flag). Soft signal, not auto-fail.
Plus deterministic numeric integrity check against the full PDF text.

Salience: imported lazily to avoid the eval ↔ agents.phases ↔ grade cycle.

Semantic-scorer dependencies (torch + sentence-transformers) are bundled in
the default install. The first scoring call downloads ~133 MB of model
weights (HuggingFace cache). The semantic scorer is optional at runtime —
`grade --no-semantic` skips it; the rest of the grader degrades gracefully
if the model can't be loaded.

Public API is bound lazily so the package can be imported even when not all
submodules are present (e.g. during incremental development).
"""

from __future__ import annotations


def grade_page(*args, **kwargs):
    from .fidelity.paper import grade_page as _impl
    return _impl(*args, **kwargs)


def parse_claims(*args, **kwargs):
    from .parser import parse_claims as _impl
    return _impl(*args, **kwargs)


__all__ = ["grade_page", "parse_claims"]
