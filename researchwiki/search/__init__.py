"""Search backend factory + high-level `suggest_category` helper.

Public API:
  - `get_default_backend()` returns a ready-to-use SearchBackend (Tantivy).
  - `suggest_category(backend, title, abstract)` classifies a paper using
    an LLM-judged version of the kNN-evidence approach (with kNN-only fallback).
  - `build_documents_from_wiki()` walks the wiki and produces Document objects
    for the index builder.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass

from ..wiki import extract_section, read_pages
from ..index.types import (
    Document,
    SearchBackend,
    SearchBackendUnavailable,
    SearchHit,
)
from ..index.pages_bm25 import TantivySearchBackend

__all__ = [
    "Document",
    "SearchBackend",
    "SearchBackendUnavailable",
    "SearchHit",
    "Suggestion",
    "get_default_backend",
    "suggest_category",
    "suggest_category_knn",
    "suggest_category_llm",
    "build_documents_from_wiki",
    "format_claim_ref",
]


@dataclass
class Suggestion:
    category: str
    confidence: float                     # top_count / len(paper_hits)
    top_3: list[tuple[str, int]]          # [(category, count), ...] sorted desc
    strength: str = "strong"              # 'strong' (≥ min_agreement) | 'weak' (top-1 fallback)
    # True when the classifier *declined* rather than chose. `category` is then
    # "other" — but so is a deliberate `other` for a genuinely cross-cutting
    # paper, and the two are different events. Without this flag the decision is
    # unrecoverable downstream: `eval classifier` counted abstentions to `other`
    # as ordinary predictions and reported 0% abstention on a run that abstained
    # ten times.
    abstained: bool = False


def get_default_backend() -> SearchBackend:
    """Return the default search backend (Tantivy)."""
    return TantivySearchBackend()


def build_documents_from_wiki() -> list[Document]:
    """Walk `wiki/` and produce Document objects for the index builder.

    Each wiki page becomes one Document. Frontmatter fields (title, authors,
    year, type) populate the corresponding Document fields. The `## Summary`
    section is extracted separately so the index can weight it higher.
    """
    docs: list[Document] = []
    for p in read_pages():
        summary = extract_section(p.body, "Summary")
        docs.append(
            Document(
                stem=p.stem,
                category=p.category,
                page_type=p.page_type,
                title=p.str_field("title"),
                authors=p.str_field("authors"),
                year=p.year_int(),
                summary=summary,
                body=p.body,
                keywords=p.str_field("keywords"),
            )
        )
    return docs


def suggest_category_knn(
    backend: SearchBackend,
    title: str,
    abstract: str,
    k: int = 5,
    min_agreement: int = 3,
) -> Suggestion | None:
    """kNN-vote classifier (the original Phase-1 approach).

    Returns:
      - Suggestion with strength='strong' when ≥ min_agreement of top-k
        neighbors agree on a category.
      - Suggestion with strength='weak' when there's at least one hit but
        consensus is below min_agreement.
      - None when the index has no paper-type documents at all, or the
        seed text is empty.

    Known failure mode: when the wiki is class-imbalanced (e.g., 90 compbio
    vs. 8 ai) and a candidate paper shares vocabulary with the majority class
    (the "graph" problem), kNN reliably misclassifies it. `suggest_category_llm`
    addresses this; this function remains as a deterministic fallback.
    """
    seed = (title or "").strip() + "\n\n" + (abstract or "").strip()
    if not seed.strip():
        return None
    hits = backend.more_like_text(seed, limit=k, page_type="paper")
    if not hits:
        return None
    counts = Counter(h.category for h in hits)
    top_cat, top_count = counts.most_common(1)[0]
    confidence = top_count / len(hits)
    strength = "strong" if top_count >= min_agreement else "weak"
    return Suggestion(
        category=top_cat,
        confidence=confidence,
        top_3=counts.most_common(3),
        strength=strength,
    )


# Curated glosses for well-known domain categories. The *set* of categories
# the classifier targets is NOT hardcoded — it's derived at call time from the
# live `wiki/` tree via `categories.content_categories()` (the same source of
# truth `is_valid()` validates against), so a category added with
# `mkdir wiki/<cat>/` is offered to the classifier immediately and never has to
# be mirrored here. This dict only supplies optional descriptions: any live
# category without an entry is listed by its bare slug (still a valid target;
# it just lacks a hand-written gloss). Page-type dirs (synthesis/ideas/
# references) are excluded by content_categories() and must never be targets.
_CATEGORY_DESCRIPTIONS = {
    "genomics": "Genomes, GWAS, biostatistics, NGS methods (variant calling, phasing, pangenome graphs)",
    "compbio": "AI/ML applied to biology — protein/RNA structure prediction, sequence foundation models, multi-omics, sequencing data analysis",
    "cgt": "Cell and gene therapy, CRISPR-Cas, AAV, ASO, prime editing, base editing",
    "ai": "Pure CS/AI/ML — agent frameworks, LLM tooling, ML methodology, scientific-AI systems with no domain-specific biology focus",
    "single-cell": "Single-cell omics — scRNA-seq/multiome foundation models, cell atlases, reference mapping, cell-type annotation, data integration",
    "other": "Cross-cutting or miscellaneous — the abstention bucket when no category cleanly fits",
}


def build_category_rules() -> str:
    """Render the category list + the "method, not topic" rule from the live
    content categories of THIS wiki (read from the `wiki/` tree on each call).

    Derived rather than hardcoded so the classifier always offers exactly the
    categories that exist — e.g. a `single-cell` dir is targeted the moment it's
    created, no source edit required. `other` is always listed last (it's the
    abstention destination). The compbio/ai disambiguation only appears when
    both categories are live (it's meaningless otherwise).
    """
    from ..categories import content_categories
    cats = sorted(c for c in content_categories() if c != "other") + ["other"]
    lines = ["Categories (the live content categories of THIS wiki):"]
    for c in cats:
        desc = _CATEGORY_DESCRIPTIONS.get(c)
        lines.append(f"- {c}:" + (f"  {desc}" if desc else ""))
    lines += ["", "Rule: classify by **method**, not **topic**."]
    if "compbio" in cats and "ai" in cats:
        lines += [
            "The line between `compbio` and `ai`:",
            "- If removing biology from the paper would gut the contribution (e.g., AlphaFold 3, Evo 2, RhoFold+) → compbio",
            "- If the contribution is a method or system that just happens to be evaluated on biology benchmarks → ai when the system is the contribution and biology is one application; compbio when the biology evaluation is the central claim",
        ]
    return "\n".join(lines) + "\n"


def _classifier_system() -> str:
    """Build the classifier system prompt against the live category set."""
    return (
        "You classify a research paper into one wiki category. The wiki is "
        "class-imbalanced — the kNN votes from neighbor papers are *evidence*, not a "
        "verdict. A paper sharing vocabulary with the majority class but on a "
        "different methodological axis should still be classified by axis. Be "
        "willing to abstain (confidence < 0.6, category \"other\") when the paper "
        "doesn't cleanly fit any category.\n\n" + build_category_rules() + "\n"
        "Output JSON: {\"category\": \"<one of the names above>\", "
        "\"confidence\": 0.0-1.0, \"rationale\": \"one sentence\"}"
    )


def _build_classifier_prompt(
    title: str, abstract: str, neighbors: list[SearchHit]
) -> str:
    parts = [
        "# Paper to classify",
        f"Title: {title}",
        "",
        "Summary / abstract:",
        abstract[:2500],
        "",
        f"# Top {len(neighbors)} kNN neighbors (evidence — not a verdict)",
        "",
    ]
    for h in neighbors:
        parts.append(
            f"- [{h.category}] [[{h.key}]] — {(h.title or '').strip()[:120]}"
        )
    parts.extend([
        "",
        "Output JSON per the system prompt.",
    ])
    return "\n".join(parts)


def _parse_classifier_response(text: str) -> dict | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# Canonical category list lives in researchwiki.categories — single source of
# truth shared by validation, future bootstrap tools, and split suggesters.
# is_valid() derives the content-category set from the local wiki/ tree.
from ..categories import is_valid as _is_valid_category

LLM_ABSTAIN_THRESHOLD = 0.60


def suggest_category_llm(
    backend: SearchBackend,
    title: str,
    abstract: str,
    k: int = 8,
) -> Suggestion | None:
    """LLM-based classifier — uses kNN as evidence, decides via reasoning.

    Why this beats the kNN-vote baseline:
      - Robust to class imbalance (LLM doesn't vote-count)
      - Robust to vocabulary collisions ("graph" in pangenome papers vs.
        knowledge graphs vs. GraphRAG — LLM reads what the paper actually
        does, not just what it shares words with)
      - Built-in abstention: if confidence < `LLM_ABSTAIN_THRESHOLD`, returns
        a Suggestion with category="other" and strength="weak" rather than
        committing to a wrong category. `other` is the framework's structured
        "uncategorized backlog" bucket — papers there get monitored by
        `suggest-splits`, which proposes promotions when the bucket grows.

    Cost: ~$0.002 per call (one Sonnet call, ~700 input + ~50 output tokens).
    Latency: ~2–3 seconds per call.

    Falls through to None if the LLM module isn't importable, the API call
    fails, or the response can't be parsed — caller can chain to kNN fallback.
    """
    seed = (title or "").strip() + "\n\n" + (abstract or "").strip()
    if not seed.strip():
        return None

    try:
        hits = backend.more_like_text(seed, limit=k, page_type="paper")
    except SearchBackendUnavailable:
        hits = []

    try:
        from ..agents import llm
    except ImportError:
        return None

    prompt = _build_classifier_prompt(title, abstract, hits)
    try:
        resp = llm.call(
            phase="classifier",
            prompt=prompt,
            system=_classifier_system(),
        )
    except Exception:
        return None

    parsed = _parse_classifier_response(resp.text)
    if not parsed:
        return None

    category = (parsed.get("category") or "").strip().lower()
    confidence = float(parsed.get("confidence") or 0.0)

    # Build top_3 from the kNN neighbor distribution for transparency,
    # even though the LLM made the actual call.
    counts = Counter(h.category for h in hits) if hits else Counter()
    top_3 = counts.most_common(3)

    if not _is_valid_category(category):
        # LLM produced an unknown/invalid category (or a page-type dir) —
        # abstain to `other`. Never auto-creates a new category.
        return Suggestion(
            category="other", confidence=0.0,
            top_3=top_3, strength="weak", abstained=True,
        )

    if confidence < LLM_ABSTAIN_THRESHOLD:
        # LLM was unsure — abstain to `other`.
        return Suggestion(
            category="other", confidence=confidence,
            top_3=top_3, strength="weak", abstained=True,
        )

    # Confident LLM call — strong if confidence is high *and* matches the
    # kNN majority vote (corroborating evidence). Weak if the LLM disagrees
    # with kNN (LLM is right, but flag for human review).
    knn_top = top_3[0][0] if top_3 else None
    strength = "strong" if (knn_top == category and confidence >= 0.75) else "weak"
    return Suggestion(
        category=category, confidence=confidence,
        top_3=top_3, strength=strength,
    )


# Switch to disable LLM classifier (e.g., for testing or offline runs).
# Reads at function-call time so callers can override per-process.
_DISABLE_LLM_ENV = "RESEARCHWIKI_NO_LLM_CLASSIFY"


def suggest_category(
    backend: SearchBackend,
    title: str,
    abstract: str,
    k: int = 5,
    min_agreement: int = 3,
) -> Suggestion | None:
    """Top-level classifier — LLM-first with kNN fallback.

    Tries `suggest_category_llm` first. If that returns None (LLM unavailable
    / API failure / parse failure) or the env var
    `RESEARCHWIKI_NO_LLM_CLASSIFY=1` is set, falls back to the deterministic
    kNN classifier (`suggest_category_knn`).

    The function signature is unchanged so callers (`agents/promote.py`,
    `tasks/ingest.py`) don't need updates.
    """
    if os.environ.get(_DISABLE_LLM_ENV):
        return suggest_category_knn(backend, title, abstract, k=k,
                                    min_agreement=min_agreement)

    llm_result = suggest_category_llm(backend, title, abstract, k=max(k, 8))
    if llm_result is not None:
        return llm_result

    return suggest_category_knn(backend, title, abstract, k=k,
                                min_agreement=min_agreement)


# Read-only retrieval primitives (formerly `researchwiki.tools`). They sit
# here because they're the query-layer analogue of the storage-layer indexes
# in `researchwiki.index`. Kept as top-level exports so callers don't have to
# reach into `search.tools` directly.
from .tools import claim_lookup, claims_by_stem, pdf_section_search  # noqa: E402
from .refs import format_claim_ref  # noqa: E402
