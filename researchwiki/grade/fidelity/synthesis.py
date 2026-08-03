"""Synthesis / idea-page fidelity grading — does each claim hold in the paper
it *cites*?

Companion to `paper.py` in this subpackage: `paper.py` grades a paper page
against its own single PDF. A synthesis or idea page is different: every
claim cites a *different* paper (via inline `[[wikilink]]` or an academic
footnote `[^id]` whose definition resolves to a wikilink), so each claim
must be checked against the PDF(s) of the paper(s) it points at. That
per-citation routing is what catches cross-paper *misattribution* — a
number or assertion ascribed to paper X that does not appear in X —
which neither `check-grounding` (citation present?) nor the paper-page
fidelity grader (this paper vs. its own PDF) can see.

Pipeline, per claim unit (units come from `grounding.parse_units`, which also
splits prose/bullets and classifies claim-shape):

  1. Resolve the unit's citations to paper stems with a `papers/{stem}.pdf`.
     No resolvable paper PDF → `uncited` (check-grounding owns "has a citation";
     here it means the cited target isn't a gradable paper, e.g. a link to
     another synthesis page or a `*(model prior)*` unit).
  2. Retrieve the cited paper(s)' chunks for the claim text and score:
       - numeric integrity (the hard-fail signal): every number in the claim
         must appear in some cited paper — retrieved neighborhood first, that
         paper's full text as fallback. Across multiple cited papers a number
         is drift only if missing from *all* of them, so a hard failure means
         the number appears nowhere in any cited paper (true misattribution),
         not merely outside the chunks the claim retrieved. ISO dates in
         editorial headers are excluded.
       - BM25 + bi-encoder max-over-chunks retrieval score (best over cited
         papers) — advisory, uncalibrated.
       - negation parity against the union of cited evidence — advisory.

Verdicts:
  supported             numbers check out and retrieval clears the advisory floor.
  weak                  single-paper claim whose cited paper barely retrieves it
                        (low BM25 *and* low semantic). Advisory — surfaced,
                        not fatal.
  composite             ≥2 cited papers + a comparative cue ("faster than",
                        "vs", "outperforms"…). The cross-paper comparison
                        lives in neither PDF whole; we don't expect it to,
                        so retrieval floor is waived. Numeric drift still
                        applies. Advisory.
  misattributed         ≥1 number in the claim appears in NONE of the cited
                        papers' neighborhoods. Hard failure (drives exit 1).
  anchor_misattributed  Fine-grained mode only: the sentence cites
                        `[[stem#slug]]` but a numeric token in the sentence
                        is absent from that specific claim's text (though the
                        paper as a whole may contain it). Hard failure.
  uncited               no cited paper has a gradable PDF. Skipped.

Only `misattributed` and `anchor_misattributed` are hard failures. Retrieval
(BM25/semantic) is uncalibrated — `paper.py` is explicit about that — so it
never fails a build here; it only annotates `weak`. Run periodically and
tighten the floors against real pages before trusting `weak` as more than a hint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .. import grounding
from ..primitives import NUMERIC_TOKEN_RE, check_numerics, negation_mismatch
from ...paths import resolve_pdf
from ...pdf.text import extract_pdf
from ...index.pdf_chunks import build_pdf_index, query_pdf, get_chunk_embeddings, MAX_PDF_PAGES
from ...index import embeddings as semantic_mod


# Advisory retrieval floors. BM25 is unbounded and corpus-dependent; a true
# no-match scores ~0, a solid lexical hit is typically well above 1. These are
# deliberately permissive (favor silence over false `weak` flags) and marked
# provisional until calibrated on real synthesis pages — see module docstring.
BM25_FLOOR = 1.0
SEMANTIC_FLOOR = 0.35

TOPK = 10

# Footnote definition line: `[^id]: ... [[wikilink]]`. Maps a footnote id to the
# wikilink(s) in its definition so a `[^id]` reference in a claim resolves to a
# cited paper.
_FOOTNOTE_DEF_RE = re.compile(r"^[ \t]*\[\^([^\]\s]+)\]:[ \t]*(.*)$", re.MULTILINE)
_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]\s]+)\]")
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
# Claim anchor extractor (matches `[[stem#slug]]` and `[[category/stem#slug]]`,
# with optional `|alias`). Groups: 1=stem, 2=slug. The slug is restricted to
# the shape claim_graph.slug emits (2–3-char lowercase prefix, dash, hash) so
# an Obsidian heading link (`[[page#Definition]]`) isn't parsed as a claim
# anchor — mirrors grade/grounding.py's `_CLAIM_ANCHOR_RE`.
_CLAIM_ANCHOR_RE = re.compile(
    r"\[\[([^\]\|#\s]+)#([a-z0-9]{2,3}-[a-z0-9-]+)(?:\|[^\]]+)?\]\]"
)

# ISO dates (`2026-06-09`, `2026-06`) are not fidelity-checkable quantities —
# idea/synthesis pages carry them in editorial headers ("Update (2026-06-10)").
# Blank them before numeric extraction so a date fragment isn't mistaken for a
# misattributed number. A bare quantity like "2048 dimensions" is untouched.
_DATE_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")

# Comparative / contrastive cues that mark a cross-paper composite claim — one
# whose assertion spans two cited papers and so is not expected to appear whole
# in either PDF.
_COMPARATIVE_RE = re.compile(
    r"\b(faster|slower|higher|lower|larger|smaller|better|worse|cheaper|"
    r"more|less|greater|fewer|outperform\w*|exceed\w*|surpass\w*|"
    r"compared\s+to|relative\s+to|whereas|unlike|versus|vs\.?|"
    r"in\s+contrast|than)\b",
    re.IGNORECASE,
)


def _strip_for_numerics(text: str) -> str:
    """Blank citation markup and dates before numeric extraction.

    A citation carries a year in its slug — `[^lazzarotto-2025]`,
    `[[compbio/huang-2023-...]]`, `[[stem#kc-01]]` — which must NOT be counted
    as a claim quantity: papers rarely restate their own publication year, so a
    leaked slug-year reads as a misattributed number. Strip wikilinks and
    footnote refs (the citation tokens) and ISO dates (editorial headers) first;
    real quantities in the prose (`2048 dimensions`, `0.85`) are untouched.
    """
    cleaned = _WIKILINK_RE.sub(" ", text)
    cleaned = _FOOTNOTE_REF_RE.sub(" ", cleaned)
    cleaned = _DATE_RE.sub(" ", cleaned)
    return cleaned


@dataclass
class FidelityClaim:
    unit_index: int
    line_start: int
    text: str
    cited_stems: list[str]          # cited papers that have a gradable PDF
    unresolved_citations: list[str]  # citations that didn't map to a paper PDF
    best_stem: str | None           # cited paper that scored highest (BM25)
    best_bm25: float
    best_semantic: float | None
    numeric_unmatched: list[str]    # numbers in the claim found in NO cited paper
    negation_mismatch: bool
    verdict: str                    # supported|weak|composite|misattributed|anchor_misattributed|uncited
    # Fine-grained mode only:
    anchor_misattributions: list[dict] = field(default_factory=list)
    # each dict: {stem, slug, numeric_tokens_missing: [...]}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SynthesisFidelityReport:
    page_path: str
    n_claims: int                   # claim-shaped units with ≥1 resolvable citation
    n_supported: int
    n_weak: int
    n_composite: int
    n_misattributed: int
    n_uncited: int                  # claim-shaped units with no gradable cited PDF
    semantic_available: bool
    n_anchor_misattributed: int = 0  # fine-grained mode only
    claims: list[FidelityClaim] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no claim is misattributed (paper- or anchor-level)."""
        return self.n_misattributed == 0 and self.n_anchor_misattributed == 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        d["ok"] = self.ok
        return d


def _footnote_targets(page_text: str) -> dict[str, list[str]]:
    """Map footnote id → list of wikilink targets in its definition line."""
    out: dict[str, list[str]] = {}
    for m in _FOOTNOTE_DEF_RE.finditer(page_text):
        fid, body = m.group(1).strip(), m.group(2)
        links = _WIKILINK_RE.findall(body)
        if links:
            out[fid] = links
    return out


def _wikilink_to_stem(link: str) -> str:
    """`[[compbio/abramson-2024-...]]` body → `abramson-2024-...`.

    Tolerates an `|alias` suffix and a leading category path; the stem is the
    final path segment of the link target.
    """
    target = link.split("|", 1)[0].strip()
    return target.rsplit("/", 1)[-1].strip()


def _resolve_cited_stems(
    unit_text: str, footnote_targets: dict[str, list[str]]
) -> tuple[list[str], list[str]]:
    """Return (gradable_stems, unresolved) for a claim unit.

    Citations come from inline `[[wikilink]]`s and from `[^id]` footnote refs
    resolved through `footnote_targets`. A stem is gradable iff
    `papers/{stem}.pdf` exists (links to other synthesis/reference pages, or to
    papers without a local PDF, fall into `unresolved`).
    """
    links: list[str] = list(_WIKILINK_RE.findall(unit_text))
    for fid in _FOOTNOTE_REF_RE.findall(unit_text):
        links.extend(footnote_targets.get(fid, []))

    gradable: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()
    for link in links:
        stem = _wikilink_to_stem(link)
        if stem in seen:
            continue
        seen.add(stem)
        try:
            resolve_pdf(stem)
        except FileNotFoundError:
            unresolved.append(stem)
        else:
            gradable.append(stem)
    return gradable, unresolved


def _is_composite(text: str, n_cited: int) -> bool:
    """A cross-paper comparison: ≥2 cited papers and a comparative cue."""
    return n_cited >= 2 and bool(_COMPARATIVE_RE.search(text))


def _extract_anchor_pairs(unit_text: str) -> list[tuple[str, str]]:
    """Return (stem, slug) pairs from `[[stem#slug]]` anchors in the unit.

    Stem is stripped of any category prefix (`compbio/foo` → `foo`) since
    the claims table keys on the bare stem.
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _CLAIM_ANCHOR_RE.finditer(unit_text):
        stem = m.group(1).strip().rsplit("/", 1)[-1]
        slug = m.group(2).strip()
        key = (stem, slug)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _load_claim_texts_by_slug(
    pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Resolve (stem, slug) → claim text. Missing rows are silently absent.

    We look them up in a single batch query to keep the fine-grained check
    cheap. Silent on any DB failure — fine-grained gracefully degrades to
    treating every anchor as un-checkable.
    """
    if not pairs:
        return {}
    try:
        from ...db.connection import get_connection
        conn = get_connection()
    except Exception:
        return {}
    try:
        placeholders = ",".join(["(?, ?)"] * len(pairs))
        params: list[str] = []
        for s, slug in pairs:
            params.extend([s, slug])
        rows = conn.execute(
            f"SELECT paper_stem, claim_slug, text FROM claims "
            f" WHERE (paper_stem, claim_slug) IN (VALUES {placeholders})",
            params,
        ).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()
    return {(r["paper_stem"], r["claim_slug"]): r["text"] for r in rows}


def _check_anchor_misattribution(
    unit_text: str,
) -> list[dict]:
    """For each `[[stem#slug]]` anchor in the unit, list numeric tokens in the
    unit's sentence that DON'T appear in the specific claim's text.

    Returns [] when no anchors present, or when no numeric-mismatch survives.
    Each returned dict: {stem, slug, numeric_tokens_missing: [...]}.

    Dates are stripped first (same rule as paper-level check).
    """
    pairs = _extract_anchor_pairs(unit_text)
    if not pairs:
        return []
    texts = _load_claim_texts_by_slug(pairs)
    if not texts:
        return []
    # Strip citation markup + dates so slug-suffix digits (`kc-01`), citation
    # years (`[^foo-2025]`) and ISO dates don't get counted as claim numerics.
    cleaned = _strip_for_numerics(unit_text)
    unit_numbers = NUMERIC_TOKEN_RE.findall(cleaned)
    if not unit_numbers:
        return []
    out: list[dict] = []
    for stem, slug in pairs:
        claim_text = texts.get((stem, slug))
        if not claim_text:
            continue
        # A number is anchor-missing if it doesn't appear in the specific
        # claim's text. Uses the same primitive as paper-level checks so
        # tolerance (numeric normalisation) is consistent.
        _, um = check_numerics(cleaned, claim_text, "")
        if um:
            out.append({
                "stem": stem, "slug": slug,
                "numeric_tokens_missing": list(um),
            })
    return out


def _full_text(stem: str, cache: dict[str, str]) -> str:
    """Cached full-PDF text for a cited paper (numeric drift fallback)."""
    if stem not in cache:
        try:
            text, _ = extract_pdf(resolve_pdf(stem), max_pages=MAX_PDF_PAGES)
        except Exception:
            text = ""
        cache[stem] = text
    return cache[stem]


def _grade_claim(
    unit: grounding.Unit,
    footnote_targets: dict[str, list[str]],
    use_semantic: bool,
    fulltext_cache: dict[str, str],
    fine_grained: bool = False,
) -> FidelityClaim | None:
    """Grade one claim unit. Returns None for non-claim units."""
    if not unit.is_claim:
        return None

    cited, unresolved = _resolve_cited_stems(unit.text, footnote_targets)
    claim_text = unit.text

    if not cited:
        return FidelityClaim(
            unit_index=unit.index, line_start=unit.line_start, text=claim_text,
            cited_stems=[], unresolved_citations=unresolved, best_stem=None,
            best_bm25=0.0, best_semantic=None, numeric_unmatched=[],
            negation_mismatch=False, verdict="uncited",
        )

    # Retrieve each cited paper's neighborhood for this claim.
    per_paper_chunks: dict[str, str] = {}
    best_bm25 = 0.0
    best_stem: str | None = None
    best_semantic: float | None = None
    for stem in cited:
        try:
            build_pdf_index(stem)
        except Exception:
            # PDF unparseable/missing despite the file existing — treat the
            # citation as unresolved rather than aborting the page grade.
            unresolved.append(stem)
            continue
        hits = query_pdf(stem, claim_text, topk=TOPK)
        if not hits:
            per_paper_chunks[stem] = ""
            continue
        chunk_text = " ".join(h.text for h in hits)
        per_paper_chunks[stem] = chunk_text
        if hits[0].score > best_bm25:
            best_bm25 = hits[0].score
            best_stem = stem
        if use_semantic:
            chunk_texts = [h.text for h in hits]
            # Reuse the paper's pre-computed chunk embeddings (the expensive
            # part) when the retrieved chunks map cleanly onto the cache, so
            # only the short claim is embedded per call — mirrors coverage.py.
            chunk_embs = None
            cached = get_chunk_embeddings(stem)
            if cached is not None:
                cache_embs, cache_texts = cached
                idx_by_text = {t: i for i, t in enumerate(cache_texts)}
                rows = [idx_by_text.get(t) for t in chunk_texts]
                if all(r is not None for r in rows):
                    chunk_embs = cache_embs[rows]
            result = semantic_mod.score_claim(
                claim_text, chunk_texts, chunk_embeddings=chunk_embs,
            )
            if result is not None:
                best_semantic = result.score if best_semantic is None \
                    else max(best_semantic, result.score)

    graded_stems = [s for s in cited if s in per_paper_chunks]
    if not graded_stems:
        # Every cited paper failed to index — nothing to grade against.
        return FidelityClaim(
            unit_index=unit.index, line_start=unit.line_start, text=claim_text,
            cited_stems=[], unresolved_citations=unresolved, best_stem=None,
            best_bm25=0.0, best_semantic=None, numeric_unmatched=[],
            negation_mismatch=False, verdict="uncited",
        )

    # Numeric integrity (the hard-fail signal): a number is drift only if
    # unmatched in EVERY cited paper. Per paper we match against the retrieved
    # neighborhood first, then the paper's full text as fallback — so a hard
    # failure means the number appears *nowhere* in any cited paper (true
    # cross-paper misattribution), not merely outside the chunks this claim
    # happened to retrieve. Citation markup + dates are stripped first so a
    # slug-year (`[^foo-2025]`) or editorial header isn't scored as a quantity.
    cleaned = _strip_for_numerics(claim_text)
    tokens = NUMERIC_TOKEN_RE.findall(cleaned)
    unmatched_everywhere = set(tokens)
    for stem in graded_stems:
        evidence = per_paper_chunks[stem]
        _, um = check_numerics(cleaned, evidence, _full_text(stem, fulltext_cache))
        unmatched_everywhere &= set(um)
    numeric_unmatched = [t for t in tokens if t in unmatched_everywhere]

    combined_evidence = " ".join(per_paper_chunks[s] for s in graded_stems)
    neg_mismatch = negation_mismatch(claim_text, combined_evidence)

    composite = _is_composite(claim_text, len(graded_stems))

    # Fine-grained anchor check: catches specific-claim misattribution that
    # paper-level retrieval would silently pass (the number IS in the paper
    # somewhere, but not in the claim you cited).
    anchor_misses: list[dict] = []
    if fine_grained:
        anchor_misses = _check_anchor_misattribution(claim_text)

    if numeric_unmatched:
        verdict = "misattributed"
    elif anchor_misses:
        verdict = "anchor_misattributed"
    elif composite:
        verdict = "composite"
    elif (best_bm25 < BM25_FLOOR
          and (best_semantic is None or best_semantic < SEMANTIC_FLOOR)):
        verdict = "weak"
    else:
        verdict = "supported"

    return FidelityClaim(
        unit_index=unit.index, line_start=unit.line_start, text=claim_text,
        cited_stems=graded_stems, unresolved_citations=unresolved,
        best_stem=best_stem, best_bm25=best_bm25, best_semantic=best_semantic,
        numeric_unmatched=numeric_unmatched, negation_mismatch=neg_mismatch,
        verdict=verdict, anchor_misattributions=anchor_misses,
    )


def grade_synthesis(
    page_path: Path | str,
    semantic: bool = True,
    fine_grained: bool = False,
) -> SynthesisFidelityReport:
    """Grade a synthesis / idea page: each claim against the PDF(s) it cites.

    Args:
      page_path: markdown file to grade.
      semantic: also run the bi-encoder retrieval score (falls back to BM25 +
                numeric + negation if sentence-transformers is unavailable).
      fine_grained: when True, verify `[[stem#slug]]` anchors at the *specific
                claim* level — a number in the sentence that's in the paper
                but NOT in the cited claim's text triggers `anchor_misattributed`.
    """
    path = Path(page_path)
    text = path.read_text(encoding="utf-8")

    permissive = grounding._is_idea_page(text)  # idea pages: model-prior units OK
    units = grounding.parse_units(text, permissive=permissive)
    footnote_targets = _footnote_targets(text)
    use_semantic = semantic and semantic_mod.is_available()

    fulltext_cache: dict[str, str] = {}
    claims: list[FidelityClaim] = []
    for u in units:
        graded = _grade_claim(u, footnote_targets, use_semantic, fulltext_cache,
                              fine_grained=fine_grained)
        if graded is not None:
            claims.append(graded)

    def _count(v: str) -> int:
        return sum(1 for c in claims if c.verdict == v)

    n_uncited = _count("uncited")
    return SynthesisFidelityReport(
        page_path=str(path),
        n_claims=len(claims) - n_uncited,
        n_supported=_count("supported"),
        n_weak=_count("weak"),
        n_composite=_count("composite"),
        n_misattributed=_count("misattributed"),
        n_anchor_misattributed=_count("anchor_misattributed"),
        n_uncited=n_uncited,
        semantic_available=use_semantic,
        claims=claims,
    )
