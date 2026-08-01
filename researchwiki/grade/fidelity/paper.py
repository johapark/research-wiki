"""Paper-page fidelity grading — claims in a wiki paper page vs. its source PDF.

Per-claim score = top-1 BM25 over per-PDF chunks; page composite = mean of
non-cross-ref claims. Raw BM25 scores are unbounded and corpus-dependent —
we report them alongside `top3_mean` (average over top-3 chunks, smoother)
to give the calibration step enough information to design thresholds.

Numeric integrity is checked deterministically: every numeric token in the
claim must appear verbatim in the top-3 retrieved chunks, else flagged.

Companion to `synthesis.py` in this subpackage, which routes per-claim
fidelity through `[[wikilink]]` resolution to multiple cited PDFs and
produces categorical verdicts (supported/weak/composite/misattributed).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ...wiki import Page, read_page
from ...paths import wiki_dir, resolve_pdf
from ...pdf.text import extract_pdf
from ..parser import Claim, parse_claims
from ..primitives import normalize_numeric as _normalize_numeric, check_numerics as _check_numerics, negation_mismatch as _negation_mismatch
from ...index.pdf_chunks import build_pdf_index, query_pdf, get_chunk_embeddings, MAX_PDF_PAGES
from ...index import embeddings as semantic_mod
from ...pdf.sections import anchor_sections
from ..salience import SalienceReport, score_salience

try:
    from ...db.connection import get_connection as _db_connection
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False


# The three names `fidelity/__init__.py` re-exports, plus `_normalize_numeric`.
# That last one is re-exported rather than used here: it's the primitive whose
# behavior `tests/test_numeric_drift.py` pins, and the test reaches for it through
# this module because this is where the numeric-drift veto is assembled. Listing
# it makes the unused-import guard (tests/test_no_unused_imports.py) read the
# re-export as intentional rather than dead. Nothing in the repo star-imports, so
# this list is documentation, not machinery.
__all__ = [
    "ClaimScore",
    "PaperFidelityReport",
    "grade_page",
    "_normalize_numeric",
]


@dataclass
class ClaimScore:
    section: str
    position: int
    text: str
    is_cross_ref: bool
    top1_score: float
    top3_mean: float
    top1_chunk_id: int
    supporting_text: str
    numeric_tokens: list[str]
    numeric_unmatched: list[str]   # numeric tokens not found anywhere in the PDF
    semantic_score: float | None   # max cosine similarity over top-K chunks (bi-encoder); None if model unavailable
    negation_mismatch: bool        # claim has negation token; top-K chunks don't (soft contradiction signal)
    graded: bool                   # False if cross-ref (skipped)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperFidelityReport:
    stem: str
    page_path: str
    n_claims: int
    n_graded: int
    n_cross_refs: int
    mean_top1: float
    median_top1: float
    weakest_claim: str | None
    weakest_score: float | None
    n_with_numeric_drift: int
    semantic_available: bool
    semantic_score: float | None
    semantic_median: float | None
    n_negation_mismatches: int      # claims flagged by the deterministic negation parity check
    salience: SalienceReport | None  # PDF-anchor recall (None when --no-salience or zero anchors)
    claims: list[ClaimScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        return d


def _score_claim(stem: str, claim: Claim, full_pdf_text: str, use_semantic: bool) -> ClaimScore:
    if claim.is_cross_ref:
        return ClaimScore(
            section=claim.section,
            position=claim.position,
            text=claim.text,
            is_cross_ref=True,
            top1_score=0.0,
            top3_mean=0.0,
            top1_chunk_id=-1,
            supporting_text="",
            numeric_tokens=[],
            numeric_unmatched=[],
            semantic_score=None,
            negation_mismatch=False,
            graded=False,
        )

    hits = query_pdf(stem, claim.text, topk=5)
    if not hits:
        return ClaimScore(
            section=claim.section,
            position=claim.position,
            text=claim.text,
            is_cross_ref=False,
            top1_score=0.0,
            top3_mean=0.0,
            top1_chunk_id=-1,
            supporting_text="",
            numeric_tokens=[],
            numeric_unmatched=[],
            semantic_score=None,
            negation_mismatch=False,
            graded=True,
        )

    top1 = hits[0]
    top3 = hits[:3]
    top3_mean = sum(h.score for h in top3) / len(top3)

    haystack = " ".join(h.text for h in top3)
    nums, unmatched = _check_numerics(claim.text, haystack, full_pdf_text)

    # Negation parity is cheap and orthogonal to the embedding signal — always
    # run it on the same top-3 retrieved chunks the numeric check uses.
    neg_mismatch = _negation_mismatch(claim.text, haystack)

    semantic_score: float | None = None
    if use_semantic:
        # Widen retrieval to top-10 for the embedding scorer (same rationale as
        # the prior NLI path: multi-aspect claims have evidence spread across
        # multiple chunks). The bi-encoder's max-over-chunks aggregation makes
        # this safe; irrelevant chunks just don't move the max.
        sem_hits = query_pdf(stem, claim.text, topk=10)
        if sem_hits:
            chunk_texts = [h.text for h in sem_hits]
            # If the per-PDF embedding cache is populated and the chunk_id ↔
            # text mapping matches, reuse the cached embeddings for the
            # retrieved chunks. Otherwise fall back to embedding the chunks
            # ad-hoc inside score_claim — slower but correct.
            cached = get_chunk_embeddings(stem)
            chunk_embs = None
            if cached is not None:
                cache_embs, cache_texts = cached
                idx_by_text = {t: i for i, t in enumerate(cache_texts)}
                rows = [idx_by_text.get(t) for t in chunk_texts]
                if all(r is not None for r in rows):
                    chunk_embs = cache_embs[rows]
            result = semantic_mod.score_claim(
                claim.text, chunk_texts, chunk_embeddings=chunk_embs,
            )
            if result is not None:
                semantic_score = result.score

    return ClaimScore(
        section=claim.section,
        position=claim.position,
        text=claim.text,
        is_cross_ref=False,
        top1_score=top1.score,
        top3_mean=top3_mean,
        top1_chunk_id=top1.chunk_id,
        supporting_text=top1.text[:500],
        numeric_tokens=nums,
        numeric_unmatched=unmatched,
        semantic_score=semantic_score,
        negation_mismatch=neg_mismatch,
        graded=True,
    )


def _persist_scores(stem: str, scored: list[ClaimScore], *, embed_model: str | None) -> None:
    """Update grader-output columns on existing claims rows.

    The rows themselves come from `db rebuild`. We don't INSERT here — if
    a claim row is missing, that means the DB hasn't been rebuilt since the
    page was edited; let `db verify` flag it. Better to skip than to fork
    the source of truth.

    `embed_model` records which bi-encoder produced the semantic_score so the
    regression task can detect cross-model comparisons (semantic scores are
    incomparable across model swaps; BM25 stays valid).
    """
    import json as _json
    import time as _time
    if not scored:
        return
    try:
        conn = _db_connection()
    except Exception:
        return
    now = int(_time.time())
    with conn:
        for s in scored:
            conn.execute(
                """
                UPDATE claims
                SET bm25_top1 = ?,
                    bm25_top3_mean = ?,
                    bm25_top1_chunk_id = ?,
                    supporting_text = ?,
                    semantic_score = ?,
                    embed_model = ?,
                    negation_mismatch = ?,
                    numeric_tokens = ?,
                    numeric_unmatched = ?,
                    last_graded_at = ?
                WHERE paper_stem = ? AND section = ? AND position = ?
                """,
                (
                    s.top1_score if s.graded else None,
                    s.top3_mean if s.graded else None,
                    s.top1_chunk_id if s.graded and s.top1_chunk_id >= 0 else None,
                    s.supporting_text if s.graded and s.supporting_text else None,
                    s.semantic_score,
                    embed_model if s.semantic_score is not None else None,
                    1 if s.negation_mismatch else 0,
                    _json.dumps(s.numeric_tokens) if s.numeric_tokens else None,
                    _json.dumps(s.numeric_unmatched) if s.numeric_unmatched else None,
                    now,
                    stem,
                    s.section,
                    s.position,
                ),
            )
    conn.close()


def _resolve_page(stem: str) -> Page:
    """Find a wiki page by stem under any category subdirectory."""
    matches = list(wiki_dir().rglob(f"{stem}.md"))
    if not matches:
        raise FileNotFoundError(f"No wiki page found for stem={stem}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple wiki pages match stem={stem}: {matches}")
    page = read_page(matches[0])
    if page is None:
        raise RuntimeError(f"Wiki page {matches[0]} could not be parsed (bad frontmatter?).")
    return page


def grade_page(
    stem: str,
    page_path: Path | str | None = None,
    semantic: bool = True,
    persist: bool = True,
    pdf_path: Path | str | None = None,
    include_salience: bool = True,
) -> PaperFidelityReport:
    """Run coverage grader on a wiki page.

    Args:
      stem: paper stem; drives the per-PDF chunk index used for retrieval.
      page_path: optional override — grade an arbitrary wiki-shaped markdown
                 file (e.g. a hallucinated test fixture) against the stem's PDF.
                 If None, finds the canonical page under wiki/{category}/{stem}.md.
      semantic: when True, also runs the bi-encoder semantic scorer per claim.
                Falls back silently to BM25-only output if sentence-transformers
                isn't loadable.
      persist: when True (default) and grading the canonical page (no
               page_path override), writes per-claim scores back into the
               structured DB so other tools (synthesis, conversation, weak-
               page surfacing) can query them. Test-fixture runs always
               skip persistence.
      include_salience: when True (default), also run the salience scorer
               (PDF-anchor recall — does the page capture the lead-in
               sentences and figure captions of the source PDF?). Disable to
               skip the second pass when only fidelity matters.
    """
    if page_path is not None:
        path = Path(page_path)
        page = read_page(path)
        if page is None:
            raise RuntimeError(f"Wiki page {path} could not be parsed (bad frontmatter?).")
    else:
        page = _resolve_page(stem)

    claims = parse_claims(page)

    # PDF source: explicit override (used by the ingest agent on pre-ingest
    # PDFs) takes precedence over the canonical papers/{stem}.pdf location.
    resolved_pdf = Path(pdf_path) if pdf_path is not None else resolve_pdf(stem)

    # Ensure the per-PDF index exists before scoring claims
    build_pdf_index(stem, pdf_path=resolved_pdf)

    # Load full PDF text once for the numeric integrity fallback check
    full_pdf_text, _ = extract_pdf(resolved_pdf, max_pages=MAX_PDF_PAGES)

    semantic_available = semantic and semantic_mod.is_available()
    scored: list[ClaimScore] = [
        _score_claim(stem, c, full_pdf_text, use_semantic=semantic_available) for c in claims
    ]

    graded = [s for s in scored if s.graded]
    n_xref = sum(1 for s in scored if s.is_cross_ref)

    if graded:
        scores = sorted(s.top1_score for s in graded)
        mean_top1 = sum(scores) / len(scores)
        median_top1 = scores[len(scores) // 2]
        weakest = min(graded, key=lambda s: s.top1_score)
        weakest_text = weakest.text
        weakest_score = weakest.top1_score
    else:
        mean_top1 = 0.0
        median_top1 = 0.0
        weakest_text = None
        weakest_score = None

    n_drift = sum(1 for s in graded if s.numeric_unmatched)

    sem_scored = [s for s in graded if s.semantic_score is not None]
    if sem_scored:
        sem_vals = sorted(s.semantic_score for s in sem_scored)
        semantic_score = sum(sem_vals) / len(sem_vals)
        semantic_median = sem_vals[len(sem_vals) // 2]
    else:
        semantic_score = None
        semantic_median = None

    n_negation_mismatches = sum(1 for s in graded if s.negation_mismatch)

    if persist and page_path is None and _DB_AVAILABLE:
        _persist_scores(stem, scored, embed_model=semantic_mod.DEFAULT_MODEL if semantic else None)

    salience: SalienceReport | None = None
    if include_salience:
        sections = anchor_sections(full_pdf_text)
        # Salience semantic-cosine path tracks the same flag as fidelity:
        # `--no-semantic` disables both.
        salience = score_salience(
            stem, full_pdf_text, sections, page.body,
            use_semantic=semantic_available,
        )

    return PaperFidelityReport(
        stem=stem,
        page_path=str(page.path),
        n_claims=len(scored),
        n_graded=len(graded),
        n_cross_refs=n_xref,
        mean_top1=mean_top1,
        median_top1=median_top1,
        weakest_claim=weakest_text,
        weakest_score=weakest_score,
        n_with_numeric_drift=n_drift,
        semantic_available=semantic_available,
        semantic_score=semantic_score,
        semantic_median=semantic_median,
        n_negation_mismatches=n_negation_mismatches,
        salience=salience,
        claims=scored,
    )
