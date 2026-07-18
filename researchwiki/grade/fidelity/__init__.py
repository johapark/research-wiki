"""Fidelity grading — does the page faithfully reflect its source PDF(s)?

Two siblings, one per page-type, sharing scoring primitives but with
different routing models and verdict semantics:

  - `paper.py`     — paper page vs. its OWN PDF.
                     Per-claim BM25 + bi-encoder cosine + numeric integrity.
                     Continuous floats (mean BM25, semantic_score, weakest_score).
                     Hot path: agent ingest's per-draft loop, `grade regression`.
                     Top-level entry: `grade_page`.

  - `synthesis.py` — synthesis / idea page vs. its CITED PDFs.
                     Per-claim citation routing through `[[wikilink]]` and
                     `[^id]` footnote resolution; numeric integrity across all
                     cited papers; categorical verdicts
                     (supported / weak / composite / misattributed / uncited).
                     Hard-fails on misattribution.
                     Top-level entry: `grade_synthesis`.

Shared deterministic primitives live in `grade/primitives.py` (numeric,
negation) and `grade/parser.py` (claim extraction). Retrieval is via
`index/pdf_chunks.py` (BM25) and `index/embeddings.py` (bi-encoder cosine).
"""

from __future__ import annotations

from .paper import (
    ClaimScore,
    PaperFidelityReport,
    grade_page,
)
from .synthesis import (
    FidelityClaim,
    SynthesisFidelityReport,
    grade_synthesis,
)

__all__ = [
    "ClaimScore",
    "FidelityClaim",
    "PaperFidelityReport",
    "SynthesisFidelityReport",
    "grade_page",
    "grade_synthesis",
]
